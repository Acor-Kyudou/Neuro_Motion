import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import pytorch_lightning as pl
import os

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

class EEGClassificationLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EEGClassificationModel()

    def forward(self, x):
        return self.model(x)

def predict(model_path, data_path, label_path, model_format='pth'):
    # Reject foot data
    data_filename = os.path.basename(data_path).lower()
    label_filename = os.path.basename(label_path).lower()
    if data_filename == 'x_test_foot.npy' or label_filename == 'y_test_foot.npy':
        raise ValueError("Foot data (X_test_foot.npy or y_test_foot.npy) is not supported for hand simulation")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    if data_filename != 'x_test.npy' or label_filename != 'y_test.npy':
        raise ValueError("Only X_test.npy and y_test.npy are supported for hand simulation")

    data = np.load(data_path)
    labels = np.load(label_path)
    if data.shape[1:] != (64, 656):
        raise ValueError(f"Invalid data shape {data.shape}, expected (n_epochs, 64, 656)")
    data = torch.tensor(data, dtype=torch.float32)

    if model_format == 'pth':
        model = EEGClassificationModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            logits = model(data)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float().numpy().flatten()
    elif model_format == 'ckpt':
        model = EEGClassificationLightning.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            logits = model(data)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float().numpy().flatten()
    elif model_format == 'onnx':
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        data = data.numpy()
        logits = session.run(None, {input_name: data})[0]
        probabilities = 1 / (1 + np.exp(-logits))
        predictions = (probabilities > 0.5).astype(np.float32).flatten()
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

    accuracy = np.mean(predictions == labels)
    print(f"Model accuracy on test set: {accuracy:.4f}")
    return predictions

if __name__ == "__main__":
    base_path = r"C:\Users\USER\Downloads\sahan"
    model_file = r"models\model_pytorch.pth"
    model_format = "pth"
    data_file = "X_test.npy"
    label_file = "y_test.npy"
    model_path = os.path.join(base_path, model_file)
    data_path = os.path.join(base_path, data_file)
    label_path = os.path.join(base_path, label_file)
    predictions = predict(model_path, data_path, label_path, model_format)
    np.save(os.path.join(base_path, "predictions_hand.npy"), predictions)