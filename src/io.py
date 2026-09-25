import spikeinterface as si
from src.config import (
    RAW_DATA_DIR, FS_ORIGINAL, NUM_CHANNELS, DTYPE, INTERIM_DATA_DIR, EVENT_SUFFIXES,
    PROCESSED_DATA_DIR, MULTITAPER_PARAMS, EPOCH_T_PRE, FS_LFP, SESSION_METADATA,
)
import numpy as np
import pandas as pd

def inspect_behavior(subject: str, session: str) -> None:
    """
    Reads behavioral CSV files (Grasp and Steps) and prints a descriptive summary.
    Adapts dynamically to the available columns in the datasets.
    """
    events_dir = RAW_DATA_DIR / subject / session / "Events"
    steps_file = events_dir / f"{session}{EVENT_SUFFIXES['steps']}"
    grasp_file = events_dir / f"{session}{EVENT_SUFFIXES['grasp']}"
    
    print(f"=== Behavior Summary: {subject} | {session} ===\n")
    
    # 1. Grasp Analysis
    if grasp_file.exists():
        df_grasp = pd.read_csv(grasp_file)
        num_grasps = len(df_grasp)
        print("[GRASP]")
        print(f"  - Total grasps recorded: {num_grasps}")
            
        if 'Target' in df_grasp.columns and 'Hand' in df_grasp.columns:
            print("  - Breakdown by Target and Hand:")
            breakdown = df_grasp.groupby(['Target', 'Hand']).size().to_string(header=False)
            print(f"    {breakdown.replace(chr(10), chr(10) + '    ')}")
    else:
        print("[GRASP]\n  - File not found.")
        
    print("\n" + "-"*45 + "\n")
    
    # 2. Walk/Steps Analysis
    if steps_file.exists():
        df_steps = pd.read_csv(steps_file)
        print("[WALK]")
        
        num_walks = df_steps['WalkNumber'].nunique()
        total_steps = len(df_steps)
        
        print(f"  - Total walks: {num_walks}")
        print(f"  - Total individual steps: {total_steps}")
        
        if num_walks > 0:
            # Calculate mean steps per walk (total steps / unique walks)
            mean_steps = total_steps / num_walks
            print(f"  - Mean steps per walk: {mean_steps:.1f}")
            
            # Calculate walk duration: max(StepTime) - min(StepTime) for each WalkNumber
            walk_durations = df_steps.groupby('WalkNumber')['StepTime'].agg(lambda x: x.max() - x.min())
            mean_walk_dur = walk_durations.mean()
            print(f"  - Mean walk duration: {mean_walk_dur:.2f} s")
            
            # Breakdown by Surface
            if 'Surface' in df_steps.columns:
                print("  - Walks breakdown by Surface:")
                # Group by walk and take the first surface to avoid counting every step
                surface_counts = df_steps.groupby('WalkNumber')['Surface'].first().value_counts()
                for surface, count in surface_counts.items():
                    print(f"    - {surface}: {count} walks")

            # Breakdown by Hand
            if 'Hand' in df_steps.columns:
                print("  - Steps breakdown by Hand:")
                hand_counts = df_steps['Hand'].value_counts()
                for hand, count in hand_counts.items():
                    print(f"    - {hand}: {count} steps")
    else:
        print("[WALK]\n  - File not found.")

def load_binary_session(subject: str, session: str) -> si.core.BinaryRecordingExtractor:
    """
    Loads a multiplexed binary file into SpikeInterface.
    
    Parameters
    ----------
    subject : str
        Subject name (e.g., 'Router').
    session : str
        Session name (e.g., 'Router_20220211').
        
    Returns
    -------
    recording : sc.BinaryRecordingExtractor
        SpikeInterface recording object ready for preprocessing.
    """
    # Construct the exact path mapped by the MATLAB script
    bin_path = RAW_DATA_DIR / subject / session / 'Wideband' / f"{session}_raw.bin"
    
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found at {bin_path}")

    # Load into SpikeInterface as interleaved (C-order by default)
    recording = si.core.read_binary(
        file_paths=bin_path,
        sampling_frequency=FS_ORIGINAL,
        num_channels=NUM_CHANNELS,
        dtype=DTYPE, 
        is_filtered=False
    )
    
    return recording

def load_lfp_recording(subject: str, session: str, folder_name: str) -> si.core.BaseRecording:
    """
    Load a SpikeInterface recording object previously saved to disk.
    """
    recording_path = INTERIM_DATA_DIR / subject / session / folder_name

    if not recording_path.exists():
        raise FileNotFoundError(f"Saved recording not found at {recording_path}")

    return si.load(recording_path)

def load_multitaper_epochs(subject: str, session: str, event_type: str) -> dict:
    """
    Loads one stage-3 .npz for (subject, session, event_type) and rebuilds the trial
    metadata that T-07 stops duplicating on disk, by joining the .npz back to the event
    CSV on event time. This is the only function that should read this .npz file.

    Returns a dict with the keys stored by src.preprocessing.extract_multitaper_epochs
    ('mt_tensor', 'event_times_s', 'freqs', 'normalization', 'robust', 'mu', 'sigma',
    'n_reference_windows', 'normalization_seed', 'reference_centers_s',
    'excluded_artifact_times_s'), plus:
        times_s : (n_times,) array, the epoch time axis in seconds (0 at the event).
        trials : DataFrame, one row per tensor trial in the same order as 'mt_tensor',
            with the original event CSV columns plus:
              - 'is_contralateral': Hand compared against the hemisphere in
                SESSION_METADATA (the only place this comparison is made).
              - 'action_class': one of 'steps', 'grasp_hook', 'grasp_floor'.
              - 'is_first_or_last_in_walk' (event_type == 'steps' only): True for the
                first or last step of its WalkNumber, computed on the full event table
                before stage 3's artefact/boundary exclusions, so a step that was the
                true first of its walk is not miscounted just because an earlier step
                in that walk happened to be dropped.
    """
    target_fs = MULTITAPER_PARAMS['target_fs']
    file_path = PROCESSED_DATA_DIR / subject / session / f"epoched_multitaper_{event_type}_{int(target_fs)}Hz_{int(MULTITAPER_PARAMS['window_taper_s']*1000)}ms.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Multitaper epoched data not found at {file_path}")

    with np.load(file_path, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}

    mt_tensor = result['mt_tensor']
    event_times_s = result['event_times_s']

    # 1. Load the full event CSV and compute row-level properties BEFORE the join, so
    # trials dropped by stage 3 (artefact mask, epoch boundary) cannot affect them.
    events_dir = RAW_DATA_DIR / subject / session / "Events"
    csv_file = events_dir / f"{session}{EVENT_SUFFIXES[event_type]}"
    df_events = pd.read_csv(csv_file)

    if event_type == 'grasp':
        time_col = 'EventTime'
        unrecognized = ~df_events['Target'].isin(['hook', 'floor'])
        if unrecognized.any():
            raise ValueError(
                f"Unrecognized Target value(s) in {csv_file}: "
                f"{df_events.loc[unrecognized, 'Target'].unique().tolist()}"
            )
        df_events['action_class'] = np.where(df_events['Target'] == 'hook', 'grasp_hook', 'grasp_floor')
    elif event_type == 'steps':
        time_col = 'StepTime'
        df_events['action_class'] = 'steps'
        walk_extent = df_events.groupby('WalkNumber')[time_col].agg(['min', 'max'])
        is_min = df_events[time_col].values == df_events['WalkNumber'].map(walk_extent['min']).values
        is_max = df_events[time_col].values == df_events['WalkNumber'].map(walk_extent['max']).values
        df_events['is_first_or_last_in_walk'] = is_min | is_max
    else:
        raise ValueError(f"Unknown event_type {event_type!r}, expected 'grasp' or 'steps'.")

    df_events['event_time_s'] = df_events[time_col]

    hemisphere = SESSION_METADATA.loc[session, 'Hemisphere']
    contralateral_hand = 'R' if hemisphere == 'L' else 'L'
    df_events['is_contralateral'] = df_events['Hand'] == contralateral_hand

    # 2. Join event_times_s (the trials stage 3 kept, in tensor order) back onto the full
    # event table, matching within half a sample at the LFP rate.
    tolerance_s = 0.5 / FS_LFP
    event_time_values = df_events['event_time_s'].values
    row_indices = []
    for t in event_times_s:
        matches = np.flatnonzero(np.abs(event_time_values - t) <= tolerance_s)
        if matches.size == 0:
            raise ValueError(
                f"event_times_s value {t} has no match in {csv_file} within "
                f"{tolerance_s * 1000:.3f} ms."
            )
        if matches.size > 1:
            raise ValueError(
                f"event_times_s value {t} matches {matches.size} rows in {csv_file} "
                f"within {tolerance_s * 1000:.3f} ms (expected exactly one)."
            )
        row_indices.append(matches[0])
    trials = df_events.iloc[row_indices].reset_index(drop=True)

    # 3. Time axis (T-08 item 4): built once, here, from config, instead of being
    # reconstructed independently by every caller.
    num_times = mt_tensor.shape[2]
    times_s = np.arange(num_times) / target_fs - EPOCH_T_PRE
    assert len(times_s) == mt_tensor.shape[2], "times_s length does not match the tensor's time axis"

    result['trials'] = trials
    result['times_s'] = times_s
    return result
