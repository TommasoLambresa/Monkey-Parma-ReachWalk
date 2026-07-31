from pathlib import Path

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
}

STATISTICAL_PARAMS = {
    'alpha': 0.05,  # Significance level for ANOVA and post-hoc tests
    'bin_size_ms': 100,  # Size of time bins for analysis
}

