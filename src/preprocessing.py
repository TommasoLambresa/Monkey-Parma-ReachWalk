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
    
    # 1. Bandpass filter (1-250 Hz)
    # ignore_low_freq_error=True bypasses the low frequency safety check in SI.
    # The margin is automatically set to 5 seconds.
    recording_bp = spr.bandpass_filter(
        recording_uV, 
        freq_min=1.0, 
        freq_max=250.0, 
        ignore_low_freq_error=True 
    )

    
    # 2. Downsample to an intermediate frequency (e.g., 1000 Hz)
    # Respects Nyquist theorem for the high_gamma band (250 Hz)
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

def extract_multitaper_epochs(subject: str, session: str, event_type: str = 'grasp') -> None:
    """
    Extracts epoched LFP data and computes Multitaper Spectrogram on padded windows.
    Saves the 4D tensor (trials, freqs, time, channels) to an .npz file.
    """

    events_dir = RAW_DATA_DIR / subject / session / "Events"
    csv_file = events_dir / f"{session}{EVENT_SUFFIXES.get(event_type)}"
    target_fs = MULTITAPER_PARAMS['target_fs'] 
    pad_s = MULTITAPER_PARAMS['pad_s']

    if not csv_file or not csv_file.exists():
        print(f"No '{event_type}' events found for {subject}/{session}.")
        return

    df_events = pd.read_csv(csv_file)
    
    # Construct timestamps and labels (No baseline_timestamps needed anymore!)
    if event_type == 'grasp':
        timestamps = df_events['EventTime'].values
        labels = df_events['Target'].fillna('unknown').astype(str) + "_" + df_events['Hand'].fillna('unknown').astype(str)
        labels = labels.values
    elif event_type == 'steps':
        timestamps = df_events['StepTime'].values
        labels = df_events['StepType'].fillna('unknown').astype(str) + "_" + \
                 df_events['Hand'].fillna('unknown').astype(str) + "_" + \
                 df_events['Surface'].fillna('unknown').astype(str)
        labels = labels.values

    # Apply manual artifact mask
    bad_trials_file = PROCESSED_DATA_DIR / subject / session / f"bad_trials_{event_type}.csv"
    if bad_trials_file.exists():
        df_bad = pd.read_csv(bad_trials_file)
        valid_mask = ~df_bad['is_artifact'].values.astype(bool)
        timestamps = timestamps[valid_mask]
        labels = labels[valid_mask]

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
    
    epoched_multitaper = []
    valid_labels_pass1 = []

    for t, label in zip(tqdm(timestamps, desc=f"Processing {event_type} Multitaper", total=len(timestamps)), labels):
        idx = int(t * fs_lfp)
        
        # Boundary check
        if (idx - samples_pre_pad < 0) or (idx + samples_post_pad > total_samples):
            continue

        padded_epoch = recording_lfp.get_traces(
            start_frame= idx - samples_pre_pad, 
            end_frame= idx + samples_post_pad, 
            return_scaled=False
        )
        data_mne_epoch = padded_epoch.T[np.newaxis, :, :]
        power_epoch = tfr_array_multitaper(
            data_mne_epoch, sfreq=fs_lfp, freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, output='power', n_jobs=1, decim=ds_factor  
        )[0].transpose(1, 2, 0)
        
        power_epoch_db = 10 * np.log10(power_epoch)
        
        smoothing_window = int(MULTITAPER_PARAMS['smoothing_window_s'] * fs_lfp / ds_factor)
        
        # Clipping the padded edges to avoid edge artifacts
        power_epoch_db = uniform_filter1d(power_epoch_db, size=smoothing_window, axis=1)
        power_epoch_trimmed = power_epoch_db[:, pad_ds:-pad_ds, :]
        
        # Trial-specific Z-score normalization 
        ep_mean = np.mean(power_epoch_trimmed, axis=1, keepdims=True)
        ep_std = np.std(power_epoch_trimmed, axis=1, keepdims=True)
        ep_std = np.where(ep_std == 0, 1.0, ep_std)
        
        trial_norm = (power_epoch_trimmed - ep_mean) / ep_std
        
        epoched_multitaper.append(trial_norm)
        valid_labels_pass1.append(label)

    if not epoched_multitaper:
        print("No valid epochs extracted.")
        return
        
    epoched_arr = np.stack(epoched_multitaper)
    valid_labels_arr = np.array(valid_labels_pass1)

    # Save output
    out_folder = PROCESSED_DATA_DIR / subject / session
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"epoched_notouch_multitaper_{event_type}_{int(target_fs)}Hz_{int(MULTITAPER_PARAMS['window_taper_s']*1000)}ms.npz"
    
    np.savez_compressed(
        out_path, 
        mt_tensor=epoched_arr, 
        labels=valid_labels_arr,
        freqs=freqs
    )
    print(f"Saved Multitaper tensor {epoched_arr.shape} to {out_path}")