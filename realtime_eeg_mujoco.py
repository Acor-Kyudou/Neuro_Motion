import torch
import torch.nn as nn
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
                0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = nn.functional.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = nn.functional.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x

class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel=64, dropout=0.125):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(eeg_channel, eeg_channel, 11, 1, padding=5, bias=False),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False),
            nn.BatchNorm1d(eeg_channel * 2),
        )
        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
        return x

def realtime_simulation(model_path, data_path, label_path):
    # Validate file paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Load model
    model = EEGClassificationModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load data
    data = np.load(data_path)  # Shape: (n_epochs, 64, 656)
    labels = np.load(label_path).flatten()  # Shape: (n_epochs,)
    print(f"Loaded {len(data)} epochs from X_test.npy")

    # Initialize MuJoCo
    mujoco_path = r"C:\Users\USER\Downloads\sahan\mujoco_menagerie-main\mujoco_menagerie-main\shadow_hand\scene_right.xml"
    if not os.path.exists(mujoco_path):
        raise FileNotFoundError(f"MuJoCo model not found: {mujoco_path}")
    mj_model = mujoco.MjModel.from_xml_path(mujoco_path)
    mj_data = mujoco.MjData(mj_model)

    # Keyframes
    open_hand = np.zeros(20)  # "open hand" from keyframes.xml
    default_pose = np.zeros(20)  # Default: initial state

    # Track accuracy
    correct = 0
    total = 0

    # Real-time simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for i in range(len(data)):
            # Prepare single epoch
            epoch = data[i:i+1]  # Shape: (1, 64, 656)
            epoch_tensor = torch.tensor(epoch, dtype=torch.float32)

            # Predict
            with torch.no_grad():
                logit = model(epoch_tensor)
                prob = torch.sigmoid(logit)
                pred = (prob > 0.5).float().item()

            # Evaluate
            if i < len(labels):
                correct += (pred == labels[i])
                total += 1

            # Update hand pose
            target = open_hand if pred == 1 else default_pose
            mj_data.ctrl[:] = target
            print(f"Epoch {i+1}/{len(data)}, Prediction: {pred}, Pose: {'Open Hand' if pred == 1 else 'Default'}")

            # Simulate for 1 second
            start_time = time.time()
            while time.time() - start_time < 1.0:
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                time.sleep(0.001)

        # Print final accuracy
        if total > 0:
            accuracy = correct / total
            print(f"Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    base_path = r"C:\Users\USER\Downloads\sahan"
    model_path = os.path.join(base_path, "models", "model_pytorch.pth")
    data_path = os.path.join(base_path, "X_test.npy")
    label_path = os.path.join(base_path, "y_test.npy")
    realtime_simulation(model_path, data_path, label_path)