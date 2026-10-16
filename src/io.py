import spikeinterface as si
from src.config import (
    RAW_DATA_DIR, FS_ORIGINAL, NUM_CHANNELS, DTYPE, INTERIM_DATA_DIR, EVENT_SUFFIXES,
    PROCESSED_DATA_DIR, MULTITAPER_PARAMS, EPOCH_T_PRE, FS_LFP, SESSION_METADATA,
)
import numpy as np
import pandas as pd

def inspect_behavior() -> pd.DataFrame:
    """
    Counts the behavioural events of every session in SESSION_METADATA, split by action
    class and by hand, in a single table covering all subjects.

    Counts are read from the event CSVs, before extract_multitaper_epochs drops trials for
    artefacts or epoch boundaries, so they are an upper bound on the trials available to
    the analyses.

    Action classes:
        grasp_hook, grasp_floor : grasps split by 'Target'.
        first_step : the step with the earliest 'StepTime' of each 'WalkNumber', whatever
            the hand (one per walk; same definition as 'is_first_in_walk' in
            load_multitaper_epochs). Steps without a 'WalkNumber' are never counted here.
        steps : every step event, first and last of each walk included.

    Hand is expressed relative to the implanted hemisphere in SESSION_METADATA:
    'contra' is the hand opposite to it, 'ipsi' the hand on the same side.

    Returns
    -------
    summary : DataFrame
        Rows indexed by (Subject, Session): one row per session, one (Subject, 'Total')
        row after the sessions of each subject and a final ('All', 'Total') row.
        Columns are a two-level index: ('Hemisphere', '') with the implanted hemisphere
        (empty on total rows), then (action_class, hand) with the event counts.
        If an event CSV is missing, its classes are empty (<NA>) for that session, the
        totals skip it, and a message is printed.
    """
    class_names = ['grasp_hook', 'grasp_floor', 'first_step', 'steps']
    hand_labels = ['contra', 'ipsi']
    count_columns = pd.MultiIndex.from_product([class_names, hand_labels])

    session_keys = []
    session_counts = []
    for session, session_info in SESSION_METADATA.iterrows():
        subject = session_info['Subject']
        contralateral_hand = 'R' if session_info['Hemisphere'] == 'L' else 'L'
        events_dir = RAW_DATA_DIR / subject / session / "Events"
        class_events = {}

        grasp_file = events_dir / f"{session}{EVENT_SUFFIXES['grasp']}"
        if grasp_file.exists():
            df_grasp = pd.read_csv(grasp_file)
            unrecognized = ~df_grasp['Target'].isin(['hook', 'floor'])
            if unrecognized.any():
                raise ValueError(
                    f"Unrecognized Target value(s) in {grasp_file}: "
                    f"{df_grasp.loc[unrecognized, 'Target'].unique().tolist()}"
                )
            class_events['grasp_hook'] = df_grasp[df_grasp['Target'] == 'hook']
            class_events['grasp_floor'] = df_grasp[df_grasp['Target'] == 'floor']
        else:
            print(f"[{subject} | {session}] Grasp file not found: {grasp_file}")

        steps_file = events_dir / f"{session}{EVENT_SUFFIXES['steps']}"
        if steps_file.exists():
            df_steps = pd.read_csv(steps_file)
            n_unassigned = int(df_steps['WalkNumber'].isna().sum())
            if n_unassigned > 0:
                print(
                    f"[{subject} | {session}] {n_unassigned} step(s) without WalkNumber: "
                    f"counted in 'steps', never in 'first_step'."
                )
            walk_start_s = df_steps.groupby('WalkNumber')['StepTime'].transform('min')
            class_events['first_step'] = df_steps[df_steps['StepTime'] == walk_start_s]
            class_events['steps'] = df_steps
        else:
            print(f"[{subject} | {session}] Steps file not found: {steps_file}")

        counts = pd.Series(pd.NA, index=count_columns, dtype='Int64')
        for class_name, df_class in class_events.items():
            unrecognized_hands = ~df_class['Hand'].isin(['L', 'R'])
            if unrecognized_hands.any():
                raise ValueError(
                    f"Unrecognized Hand value(s) in {session} ({class_name}): "
                    f"{df_class.loc[unrecognized_hands, 'Hand'].unique().tolist()}"
                )
            n_contra = int((df_class['Hand'] == contralateral_hand).sum())
            counts[(class_name, 'contra')] = n_contra
            counts[(class_name, 'ipsi')] = len(df_class) - n_contra

        session_keys.append((subject, session))
        session_counts.append(counts)

    index_names = ['Subject', 'Session']
    all_counts = pd.DataFrame(
        session_counts, index=pd.MultiIndex.from_tuples(session_keys, names=index_names)
    )

    # Each subject's sessions followed by its total, then the total over all subjects.
    blocks = []
    for subject, subject_counts in all_counts.groupby(level='Subject', sort=False):
        blocks.append(subject_counts)
        blocks.append(pd.DataFrame(
            [subject_counts.sum(min_count=1)],
            index=pd.MultiIndex.from_tuples([(subject, 'Total')], names=index_names),
        ))
    blocks.append(pd.DataFrame(
        [all_counts.sum(min_count=1)],
        index=pd.MultiIndex.from_tuples([('All', 'Total')], names=index_names),
    ))
    summary = pd.concat(blocks).astype('Int64')

    hemisphere = summary.index.get_level_values('Session').map(SESSION_METADATA['Hemisphere'])
    summary.insert(0, ('Hemisphere', ''), hemisphere.fillna(''))
    return summary

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
    Loads one multitaper .npz for (subject, session, event_type) and rebuilds the trial
    metadata by joining the .npz to the event CSV on event time. This is the only function
    that should read this .npz file.

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
                before the artefact/boundary exclusions of extract_multitaper_epochs, so a step that
                was the true first of its walk is not miscounted just because an earlier step
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
    # trials dropped by extract_multitaper_epochs (artefact mask, epoch boundary) cannot
    # affect them.
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

    # 2. Join event_times_s (the trials extract_multitaper_epochs kept, in tensor order)
    # back onto the full event table, matching within half a sample at the LFP rate.
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

    # 3. Time axis: built once, here, from config, so every caller gets the same one.
    num_times = mt_tensor.shape[2]
    times_s = np.arange(num_times) / target_fs - EPOCH_T_PRE
    assert len(times_s) == mt_tensor.shape[2], "times_s length does not match the tensor's time axis"

    result['trials'] = trials
    result['times_s'] = times_s
    return result
