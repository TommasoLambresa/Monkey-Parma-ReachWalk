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
FS_LFP = 300.0    
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
EPOCH_T_PRE = 1.0   # Seconds before the event
EPOCH_T_POST = 2.0  # Seconds after the event
# Baseline period relative to the event (for Z-score normalization or ERD/ERS computation)
BASELINE_T_START = -1.0
BASELINE_T_END = -0.5

# Frequency bands of interest for LFP
FREQ_BANDS = {
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta': [15, 30],
    'low_gamma': [30, 70],
    'high_gamma': [70, 150]
}