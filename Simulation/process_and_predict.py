import mne
import numpy as np
import onnxruntime as ort
from scipy import signal
import os
import glob

def process_and_predict(edf_files, model_path, output_dir="C:/Users/USER/Downloads/sahan/processed_data"):
    sfreq = 160
    tmin = 1.0
    tmax = 4.1
    expected_time_samples = int(4.1 * sfreq)  # 656 samples

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parts = []
    for edf_file in edf_files:
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            if raw.info['sfreq'] != sfreq:
                raw.resample(sfreq, npad="auto")
            parts.append(raw)
        except Exception as e:
            print(f"Error loading {edf_file}: {str(e)}")
            continue

    if not parts:
        raise ValueError("No EDF files loaded successfully.")

    # Concatenate raw files
    raw = mne.concatenate_raws(parts)
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    if len(eeg_channel_inds) != 64:
        raise ValueError(f"Expected 64 EEG channels, got {len(eeg_channel_inds)}")

    # Extract events
    events, annot_dict = mne.events_from_annotations(raw)
    event_map = {'T1': 2, 'T2': 3}  # left=2, right=3
    valid_events = [ev for ev in events if ev[2] in event_map.values()]
    if not valid_events:
        raise ValueError("No valid events (T1 or T2) found.")

    # Create epochs
    epoched = mne.Epochs(
        raw,
        valid_events,
        event_id={'left': 2, 'right': 3},
        tmin=tmin,
        tmax=tmax,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Get data and labels
    X = epoched.get_data() * 1e3  # Convert to millivolts
    y = epoched.events[:, 2] - 2  # 0=left, 1=right

    # Resample if needed
    current_time_samples = X.shape[-1]
    if current_time_samples != expected_time_samples:
        print(f"Resampling from {current_time_samples} to {expected_time_samples} samples")
        X_resampled = np.zeros((X.shape[0], X.shape[1], expected_time_samples), dtype=np.float32)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_resampled[i, j] = signal.resample(X[i, j], expected_time_samples)
        X = X_resampled

    X = X.astype(np.float32)
    y = y.astype(np.int64)
    print(f"Processed X shape: {X.shape}, y shape: {y.shape}")

    # Save tensors
    np.save(os.path.join(output_dir, "X_test.npy"), X)
    np.save(os.path.join(output_dir, "y_test.npy"), y)
    print(f"Saved tensors to {output_dir}/X_test.npy and {output_dir}/y_test.npy")

    # ONNX inference
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model input shape: {input_shape}")

    predictions = []
    for i in range(X.shape[0]):
        x = X[i:i+1]  # Shape: [1, 64, 656]
        print(f"Input shape for sample {i}: {x.shape}")
        ort_inputs = {input_name: x}
        y_hat = session.run(None, ort_inputs)[0]
        prob = 1 / (1 + np.exp(-y_hat))  # Sigmoid
        pred = (prob > 0.5).astype(np.int32).flatten()
        predictions.append(pred[0])

    predictions = np.array(predictions)
    print(f"Predictions shape: {predictions.shape}")

    return predictions, y

if __name__ == "__main__":
    edf_files = glob.glob(os.path.join("PhysioNet/S080/*.edf"))
    model_path = "models/model.onnx"
    predictions, labels = process_and_predict(edf_files, model_path)
    print(f"Predictions: {predictions[:10]}")
    print(f"Labels: {labels[:10]}")