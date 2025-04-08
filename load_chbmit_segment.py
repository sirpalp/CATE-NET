import mne
import numpy as np

def load_chbmit_segment(edf_path, start_time=0, duration=1.0):
    """
    Loads a 1-second EEG segment from a CHB-MIT .edf file.

    Parameters:
        edf_path : str
            Path to the .edf file
        start_time : float
            Start time in seconds (default: 0)
        duration : float
            Duration in seconds (default: 1.0)

    Returns:
        raw_segment : mne.io.RawArray
            MNE Raw object containing the EEG segment
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    start_sample = int(start_time * sfreq)
    stop_sample = start_sample + int(duration * sfreq)
    segment_data = raw._data[:, start_sample:stop_sample]
    info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types='eeg')
    segment = mne.io.RawArray(segment_data, info, verbose=False)
    return segment