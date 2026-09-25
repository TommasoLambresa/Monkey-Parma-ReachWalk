from pathlib import Path
import pandas as pd

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
# Absolute path to OneDrive for raw data
RAW_DATA_DIR = Path(r"C:\Users\tommy\OneDrive - Scuola Superiore Sant'Anna\Monkeys Parma\raw_binary")
# Local repository paths for intermediate and processed data
INTERIM_DATA_DIR = RAW_DATA_DIR.parent /'interim'
PROCESSED_DATA_DIR = RAW_DATA_DIR.parent / 'processed'

# Results directory
RESULTS_DIR = BASE_DIR / 'results'

# Ensure local output directories exist
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Session registry: single source of truth for every session-level property
# (hemisphere, channel count, ...). Nothing may be inferred from a session name instead.
SESSION_METADATA_PATH = RAW_DATA_DIR.parent / 'session_metadata.csv'
if not SESSION_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Session registry not found at {SESSION_METADATA_PATH}. See CLAUDE.md section 2."
    )
SESSION_METADATA = pd.read_csv(SESSION_METADATA_PATH).set_index('Session')
if SESSION_METADATA.index.duplicated().any():
    dupes = SESSION_METADATA.index[SESSION_METADATA.index.duplicated()].unique().tolist()
    raise ValueError(f"Duplicate Session entries in {SESSION_METADATA_PATH}: {dupes}")
if not SESSION_METADATA['Hemisphere'].isin(['L', 'R']).all():
    bad = SESSION_METADATA.loc[~SESSION_METADATA['Hemisphere'].isin(['L', 'R'])]
    raise ValueError(f"Unknown Hemisphere value(s) in {SESSION_METADATA_PATH}:\n{bad}")

# Hardware and signal parameters
FS_ORIGINAL = 32000.0  
NUM_CHANNELS = 128
FS_LFP = 1000.0
FS_ENVELOPES = 200.0    
DTYPE = 'float32'     

# Subjects and task definitions
SUBJECTS = ['Router', 'Wifi']
EVENT_SUFFIXES = {
    'steps': '_Steps.csv',
    'grasp': '_Grasp.csv'
}

GRASP_CONDITIONS = {
    'hook_L': 1,  # reach up left
    'hook_R': 2,  # reach up right
    'floor_L': 3, # reach down left
    'floor_R': 4  # reach down right
}
WALK_CONDITIONS = {
    'step_start': 5, # Mapping for the beginning of a walking sequence
    'step_end': 6    # Mapping for the end of a sequence
}

# Epoching parameters
EPOCH_T_PRE = 0.8   # Seconds before the event
EPOCH_T_POST = 0.5  # Seconds after the event

# Frequency bands of interest for LFP
FREQ_BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 7.0),
    'alpha': (7.0, 13.0),
    'beta': (13.0, 35.0),
    'gamma': (40.0, 200.0),
}

# Multitaper parameters for time-frequency analysis
MULTITAPER_PARAMS = {
    'time_bandwidth': 3.0,  # 2 tapers     
    'initial_frequency': 1.0, 
    'final_frequency': 200.0,
    'frequency_step': 2.0,
    'target_fs': 200.0, 
    'window_taper_s': 0.500,
    'pad_s': 2.0,
    'smoothing_window_s': 0.05,  # 50 ms smoothing window
    'N_REFERENCE_WINDOWS': 500,  # windows used to estimate the session-level mu/sigma (see extract_multitaper_epochs)
    'NORMALIZATION_SEED': 42,  # fixes the reference-window draw so it is identical across event types of a session
}

STATISTICAL_PARAMS = {
    'alpha': 0.05,  # Significance level for ANOVA and post-hoc tests
    'bin_size_ms': 100,  # Size of time bins for analysis
}

