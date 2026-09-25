import spikeinterface.preprocessing as spr
from src.config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EVENT_SUFFIXES, FS_LFP,
    EPOCH_T_PRE, EPOCH_T_POST, MULTITAPER_PARAMS
)
from src.io import load_binary_session, load_lfp_recording
from scipy.ndimage import uniform_filter1d
import numpy as np
import pandas as pd
from tqdm import tqdm
from mne.time_frequency import tfr_array_multitaper

def extract_and_save_lfp(subject, session, n_jobs=1):
    """
    Extracts Local Field Potentials (LFPs) following SpikeInterface best practices.
    Saves intermediate results to disk for efficiency.
    """
    recording = load_binary_session(subject, session)
    recording_uV = spr.scale(recording, gain=1e6)
    
    # 1. Bandpass filter (1-300 Hz)
    # ignore_low_freq_error=True bypasses the low frequency safety check in SI.
    # The margin is automatically set to 5 seconds.
    recording_bp = spr.bandpass_filter(
        recording_uV, 
        freq_min=1.0, 
        freq_max=300.0, 
        ignore_low_freq_error=True 
    )

    
    # 2. Downsample to an intermediate frequency (e.g., 1000 Hz)
    # Respects Nyquist theorem for the high_gamma band (300 Hz)
    recording_resampled = spr.resample(recording_bp, resample_rate=int(round(FS_LFP)))

    # 3. Apply Common Median Reference (CMR) at 1000 Hz
    recording_cmr = spr.common_reference(
        recording_resampled, 
        reference='global', 
        operator='median'
    )

    # 4. SAVE TO DISK (Crucial step required by documentation)
    # Use 30-second chunks to minimize the 5-second margin overhead.
    lfp_folder = INTERIM_DATA_DIR / subject / session / f"lfp_{int(FS_LFP)}Hz"
    
    # If the folder exists, load directly to avoid recomputing
    if lfp_folder.exists():
        print(f"LFP for {subject}/{session} already existing, cancel the data to overwrite.")
    else:
        print(f"Computing and saving LFP in 30s chunks. Please wait...")
        recording_cmr.save(
            folder=lfp_folder,
            chunk_duration="30s", # Prevents memory overload
            n_jobs=n_jobs,            # Use all available CPU cores
            progress_bar=True
        )
    
    return 

def extract_multitaper_epochs(
    subject: str,
    session: str,
    event_type: str = 'grasp',
    normalization: str = 'session',
    robust: bool = True,
) -> None:
    """
    Extracts epoched LFP data and computes Multitaper Spectrogram on padded windows.
    Saves the 4D tensor (trials, freqs, time, channels) to an .npz file.

    normalization : {'session', 'trial'}, default 'session'
        'session' z-scores every trial against a location (mu) and scale (sigma)
        estimated per channel and frequency from MULTITAPER_PARAMS['N_REFERENCE_WINDOWS']
        windows drawn at random across the whole session recording, each processed
        through the identical multitaper -> dB -> smoothing -> trimming pipeline used
        for the trial epochs. This is a normalisation against the session's overall
        power distribution, NOT a rest baseline: a stable baseline does not exist in
        this freely moving context (Lanzarini et al. 2025 make the same point for
        firing rates). Reference windows whose anchor time falls within +/- 1 s of a
        timestamp marked as artefact in any `bad_trials_<event>.csv` of this session are
        rejected. The draw depends only on the session, not on `event_type`: the RNG is
        seeded with `MULTITAPER_PARAMS['NORMALIZATION_SEED']` and the artefact exclusion
        is the union over every event type, so calling this function for 'grasp' and for
        'steps' on the same session yields bit-identical `mu`/`sigma`. This matters
        because the project compares grasp and step trials to each other: if each call
        normalised against its own slightly different reference, the mismatch would look
        like a small, constant power offset between classes and could be mistaken for a
        real difference between them.
        'trial' reproduces the legacy behaviour: each trial is z-scored independently
        against its own epoch, along the time axis.
    robust : bool, default True
        Only used when normalization == 'session'. When True, mu/sigma are the median
        and 1.4826 * MAD over the pooled reference-window samples; when False, the mean
        and standard deviation.
    """
    if normalization not in ('session', 'trial'):
        raise ValueError(f"normalization must be 'session' or 'trial', got {normalization!r}")

    events_dir = RAW_DATA_DIR / subject / session / "Events"
    csv_file = events_dir / f"{session}{EVENT_SUFFIXES.get(event_type)}"
    target_fs = MULTITAPER_PARAMS['target_fs']
    pad_s = MULTITAPER_PARAMS['pad_s']

    if not csv_file or not csv_file.exists():
        print(f"No '{event_type}' events found for {subject}/{session}.")
        return

    df_events = pd.read_csv(csv_file)

    # Only the event time is needed here. Every behavioural property (Hand, Target,
    # WalkNumber, ...) already lives in this CSV and is joined back in at load time by
    # src.io.load_multitaper_epochs (T-07) instead of being copied into the .npz.
    if event_type == 'grasp':
        timestamps = df_events['EventTime'].values
    elif event_type == 'steps':
        timestamps = df_events['StepTime'].values

    # Apply manual artifact mask for the current event type's own trials.
    bad_trials_file = PROCESSED_DATA_DIR / subject / session / f"bad_trials_{event_type}.csv"
    if bad_trials_file.exists():
        df_bad = pd.read_csv(bad_trials_file)
        valid_mask = ~df_bad['is_artifact'].values.astype(bool)
        timestamps = timestamps[valid_mask]

    # Session-level artefact exclusion for reference windows: the union over every
    # bad_trials_<event>.csv of this session (currently grasp and steps), read by
    # timestamp rather than by row position, so the excluded times do not depend on
    # which event_type is being processed and the reference draw is the same for both.
    excluded_artifact_times = []
    for bad_file in sorted((PROCESSED_DATA_DIR / subject / session).glob("bad_trials_*.csv")):
        df_bad_session = pd.read_csv(bad_file)
        excluded_artifact_times.append(
            df_bad_session.loc[df_bad_session['is_artifact'].astype(bool), 'timestamp'].values
        )
    excluded_artifact_times = np.concatenate(excluded_artifact_times) if excluded_artifact_times else np.array([])

    # Load 1000Hz LFP data
    recording_lfp = load_lfp_recording(subject, session, "lfp_1000Hz")
    fs_lfp = recording_lfp.get_sampling_frequency()
    num_channels = recording_lfp.get_num_channels()
    total_samples = recording_lfp.get_num_samples()

    freqs = np.arange(MULTITAPER_PARAMS['initial_frequency'], MULTITAPER_PARAMS['final_frequency'] + MULTITAPER_PARAMS['frequency_step'], MULTITAPER_PARAMS['frequency_step'])
    time_bandwidth = MULTITAPER_PARAMS['time_bandwidth']
    window = MULTITAPER_PARAMS['window_taper_s']

    # Dynamic window length based on frequency
    n_cycles = freqs * window
    n_cycles[n_cycles < 1.0] = 1

    # Time and downsampling parameters
    samples_pre_pad = int((EPOCH_T_PRE + pad_s) * fs_lfp)
    samples_post_pad = int((EPOCH_T_POST + pad_s) * fs_lfp)
    ds_factor = int(fs_lfp / target_fs)
    pad_ds = int(pad_s * target_fs)
    smoothing_window = int(MULTITAPER_PARAMS['smoothing_window_s'] * fs_lfp / ds_factor)

    def _multitaper_power_db(start_frame: int, end_frame: int) -> np.ndarray:
        """Run one padded window through the multitaper -> dB -> smoothing -> trim pipeline."""
        padded = recording_lfp.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=False)
        data_mne = padded.T[np.newaxis, :, :]
        power = tfr_array_multitaper(
            data_mne, sfreq=fs_lfp, freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, output='power', n_jobs=1, decim=ds_factor
        )[0].transpose(1, 2, 0)
        power_db = 10 * np.log10(power)
        power_db = uniform_filter1d(power_db, size=smoothing_window, axis=1)
        return power_db[:, pad_ds:-pad_ds, :]

    # Session-level normalization reference: mu, sigma have shape (n_freqs, n_channels).
    mu = np.array([])
    sigma = np.array([])
    n_reference_windows = 0
    reference_centers_s = np.array([])

    if normalization == 'session':
        n_reference_windows = MULTITAPER_PARAMS['N_REFERENCE_WINDOWS']
        rng = np.random.default_rng(MULTITAPER_PARAMS['NORMALIZATION_SEED'])
        min_center_idx = samples_pre_pad
        max_center_idx = total_samples - samples_post_pad
        if max_center_idx <= min_center_idx:
            raise ValueError(
                f"Recording for {subject}/{session} is too short to draw a reference "
                f"window of {(EPOCH_T_PRE + EPOCH_T_POST + 2 * pad_s):.1f} s."
            )

        reference_power = []
        reference_centers_list = []
        max_attempts = n_reference_windows * 100
        n_attempts = 0
        with tqdm(total=n_reference_windows, desc=f"Sampling reference windows ({subject}/{session})") as pbar:
            while len(reference_power) < n_reference_windows:
                n_attempts += 1
                if n_attempts > max_attempts:
                    raise RuntimeError(
                        f"Could not draw {n_reference_windows} valid reference windows for "
                        f"{subject}/{session} after {max_attempts} attempts; too much of the "
                        f"recording is marked as artefact."
                    )
                center_idx = int(rng.integers(min_center_idx, max_center_idx))
                center_t = center_idx / fs_lfp
                if excluded_artifact_times.size and np.any(np.abs(excluded_artifact_times - center_t) <= 1.0):
                    continue
                reference_power.append(_multitaper_power_db(center_idx - samples_pre_pad, center_idx + samples_post_pad))
                reference_centers_list.append(center_t)
                pbar.update(1)

        reference_centers_s = np.array(reference_centers_list)

        # (n_ref, F, T_ref, C) -> pool the reference-window and time axes together, per (freq, channel)
        reference_power = np.stack(reference_power)
        n_ref, n_freqs_ref, n_times_ref, n_ch_ref = reference_power.shape
        pooled = reference_power.transpose(1, 0, 2, 3).reshape(n_freqs_ref, n_ref * n_times_ref, n_ch_ref)

        if robust:
            mu = np.median(pooled, axis=1)
            sigma = 1.4826 * np.median(np.abs(pooled - mu[:, np.newaxis, :]), axis=1)
        else:
            mu = np.mean(pooled, axis=1)
            sigma = np.std(pooled, axis=1)
        sigma = np.where(sigma == 0, 1.0, sigma)

    epoched_multitaper = []
    event_times_pass1 = []

    for t in tqdm(timestamps, desc=f"Processing {event_type} Multitaper", total=len(timestamps)):
        idx = int(t * fs_lfp)

        # Boundary check
        if (idx - samples_pre_pad < 0) or (idx + samples_post_pad > total_samples):
            continue

        power_epoch_trimmed = _multitaper_power_db(idx - samples_pre_pad, idx + samples_post_pad)

        if normalization == 'trial':
            # Trial-specific Z-score normalization
            ep_mean = np.mean(power_epoch_trimmed, axis=1, keepdims=True)
            ep_std = np.std(power_epoch_trimmed, axis=1, keepdims=True)
            ep_std = np.where(ep_std == 0, 1.0, ep_std)
            trial_norm = (power_epoch_trimmed - ep_mean) / ep_std
        else:
            trial_norm = (power_epoch_trimmed - mu[:, np.newaxis, :]) / sigma[:, np.newaxis, :]

        epoched_multitaper.append(trial_norm)
        event_times_pass1.append(t)

    if not epoched_multitaper:
        print("No valid epochs extracted.")
        return

    epoched_arr = np.stack(epoched_multitaper)
    event_times_arr = np.array(event_times_pass1)

    # Save output
    out_folder = PROCESSED_DATA_DIR / subject / session
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"epoched_multitaper_{event_type}_{int(target_fs)}Hz_{int(MULTITAPER_PARAMS['window_taper_s']*1000)}ms.npz"

    np.savez_compressed(
        out_path,
        mt_tensor=epoched_arr,
        event_times_s=event_times_arr,
        freqs=freqs,
        normalization=normalization,
        robust=robust,
        mu=mu,
        sigma=sigma,
        n_reference_windows=n_reference_windows,
        normalization_seed=MULTITAPER_PARAMS['NORMALIZATION_SEED'],
        reference_centers_s=reference_centers_s,
        excluded_artifact_times_s=excluded_artifact_times,
    )
    print(f"Saved Multitaper tensor {epoched_arr.shape} to {out_path}")