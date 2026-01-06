import os

import torch
import torch.nn as nn
import streamlit as st


class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineLSTM, self).__init__()
        self.gestures = st.session_state.gestures

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gestures = st.session_state.gestures
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attn = TemporalAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.fc(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.gestures = st.session_state.gestures

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.fc(x)


@st.cache_resource
def load_model(device, weight_dir, model_type="baseline"):
    """
    Load a model from checkpoint.
    
    Parameters:
    -----------
    device : torch.device
        Device to load model on
    weight_dir : str
        Path to weight file
    model_type : str
        Type of model: "baseline", "bilstm", or "transformer"
    """
    input_size = 258
    hidden_size = 64
    num_classes = 30
    
    # Create model based on type
    if model_type == "baseline":
        model = BaselineLSTM(input_size, hidden_size, num_classes)
    elif model_type == "bilstm":
        model = BiLSTMWithAttention(input_size, hidden_size, num_classes)
    elif model_type == "transformer":
        model = TransformerModel(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'baseline', 'bilstm', or 'transformer'")

    # load weights into model
    if os.path.exists(weight_dir):
        try:
            loaded_model_state_dict = torch.load(weight_dir, map_location=device)
            model.load_state_dict(loaded_model_state_dict)
            print(f"Model loaded from {weight_dir}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {weight_dir} not found.")

    # set model to evaluation mode
    model.to(device)
    model.eval()
    return model