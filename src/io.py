import spikeinterface.core as sc
from pathlib import Path
from src.config import RAW_DATA_DIR, FS_ORIGINAL, NUM_CHANNELS, DTYPE

def load_binary_session(subject: str, session: str) -> sc.BinaryRecordingExtractor:
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
    recording = sc.read_binary(
        file_paths=bin_path,
        sampling_frequency=FS_ORIGINAL,
        num_channels=NUM_CHANNELS,
        dtype=DTYPE, 
        is_filtered=False
    )
    
    return recording