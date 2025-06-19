import mne
import os
import numpy as np
from scipy import signal
import torch
import torch.utils.data as data

DATASET_PATH = r"C:\Users\USER\Downloads\sahan\PhysioNet"
SAVE_PATH = r"C:\Users\USER\Downloads\sahan"
OPEN_CLOSE_LEFT_RIGHT_FIST = [3, 7, 11]
CLASSES = ["left", "right"]

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

def get_edf_paths(subject_ids, run_numbers):
    physionet_paths = []
    for subject_id in subject_ids:
        subject_folder = f"S{subject_id:03d}"
        subject_path = os.path.join(DATASET_PATH, subject_folder)
        if not os.path.exists(subject_path):
            print(f"Subject path does not exist: {subject_path}")
            continue
        for run in run_numbers:
            run_file = f"{subject_folder}R{run:02d}.edf"
            file_path = os.path.join(subject_path, run_file)
            if os.path.exists(file_path):
                physionet_paths.append(file_path)
            else:
                print(f"File does not exist: {file_path}")
    return physionet_paths

def preprocess_edf(subject_ids, run_numbers):
    print("Loading EDF files...")
    test_paths = get_edf_paths(subject_ids, run_numbers)
    print(f"Found {len(test_paths)} EDF files")

    if len(test_paths) == 0:
        raise ValueError("No EDF files found.")

    parts = []
    for path in test_paths:
        try:
            raw = mne.io.read_raw_edf(path, preload=True, stim_channel='auto', verbose='WARNING')
            sfreq = raw.info['sfreq']
            print(f"Sampling rate for {path}: {sfreq} Hz")
            if sfreq != 160:
                print(f"Resampling {path} from {sfreq} Hz to 160 Hz")
                raw.resample(160)
            parts.append(raw)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")

    if len(parts) == 0:
        raise ValueError("No EDF files were successfully loaded.")

    raw = mne.concatenate_raws(parts)
    events, annot = mne.events_from_annotations(raw)
    print(f"Annotations found: {list(annot.keys())}")
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    EEG_CHANNEL = len(eeg_channel_inds)
    print(f"Number of EEG channels: {EEG_CHANNEL}")

    # Create epochs
    epoched = mne.Epochs(
        raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
        proj=False, picks=eeg_channel_inds, baseline=None, preload=True, verbose=True
    )
    X = epoched.get_data() * 1e3  # Convert to mV
    y = epoched.events[:, 2] - 2  # Labels: 0=left, 1=right

    # Resample to match training shape
    expected_time_samples = int(4.1 * 160)  # 656 samples
    current_time_samples = X.shape[-1]
    print(f"Original X shape: {X.shape}, y shape: {y.shape}")
    if current_time_samples != expected_time_samples:
        print(f"Resampling epochs from {current_time_samples} to {expected_time_samples} samples")
        X_resampled = np.zeros((X.shape[0], X.shape[1], expected_time_samples), dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_resampled[i, j] = signal.resample(X[i, j], expected_time_samples)
        X = X_resampled
        print(f"Resampled X shape: {X.shape}")

    X = X.astype(np.float32)
    y = y.astype(np.int64)
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")

    # Class distribution
    left_count = np.sum(y == 0)
    right_count = np.sum(y == 1)
    print(f"Class distribution: Left={left_count}, Right={right_count}")

    # Save preprocessed data
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X)
    np.save(os.path.join(SAVE_PATH, 'y_test.npy'), y)
    print(f"Preprocessed data saved to: {os.path.join(SAVE_PATH, 'X_test.npy')}")
    print(f"Labels saved to: {os.path.join(SAVE_PATH, 'y_test.npy')}")

    return X, y

if __name__ == "__main__":
    subject_ids = range(80, 90)  # Example: Adjust based on your new EDF files
    preprocess_edf(subject_ids, OPEN_CLOSE_LEFT_RIGHT_FIST)