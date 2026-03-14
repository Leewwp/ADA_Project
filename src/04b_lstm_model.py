"""
LSTM Model for AQI Prediction
=============================
Implements LSTM neural network using PyTorch for time series forecasting.
Uses sequence length of 72 hours (3 days of history).
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

# Check if torch is available
try:
    HAS_TORCH = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
except ImportError:
    HAS_TORCH = False
    device = None


class AQIDataset(Dataset):
    """PyTorch Dataset for AQI time series."""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMPredictor(nn.Module):
    """LSTM model for AQI prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size // 2, 16)
        self.fc2 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM forward
        out, _ = self.lstm1(x)
        out = self.relu(out)
        out, _ = self.lstm2(out)
        out = self.relu(out)

        # Take the last time step
        out = out[:, -1, :]

        # Fully connected
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out.squeeze()


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    sequences = []
    targets = []

    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
        targets.append(y[i + seq_length])

    return np.array(sequences), np.array(targets)


def load_featured_data() -> pd.DataFrame:
    """Load featured AQI data."""
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} records")
    return df


def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str = 'aqi_target_24h',
                 seq_length: int = 72) -> Tuple:
    """Prepare data for LSTM modeling."""

    train_mask = df.index.year == 2024
    val_mask = (df.index.year == 2025) & (df.index.month <= 3)
    test_mask = (df.index.year == 2025) & (df.index.month > 3)

    train_mask = train_mask & df[target_col].notna()
    val_mask = val_mask & df[target_col].notna()
    test_mask = test_mask & df[target_col].notna()

    X_train = df.loc[train_mask, feature_cols].values
    X_val = df.loc[val_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values

    y_train = df.loc[train_mask, target_col].values
    y_val = df.loc[val_mask, target_col].values
    y_test = df.loc[test_mask, target_col].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)

    print(f"LSTM Data Shapes:")
    print(f"  Train: {X_train_seq.shape}, {y_train_seq.shape}")
    print(f"  Val:   {X_val_seq.shape}, {y_val_seq.shape}")
    print(f"  Test:  {X_test_seq.shape}, {y_test_seq.shape}")

    return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq, scaler


def train_lstm(X_train, y_train, X_val, y_val, input_size: int, epochs: int = 100,
               batch_size: int = 64, hidden_size: int = 64, lr: float = 0.001):
    """Train LSTM model."""
    if not HAS_TORCH:
        return None, None, 'lstm', None

    print(f"\n{'='*40}")
    print(f"Training LSTM (seq_len={X_train.shape[1]}, hidden={hidden_size})")
    print('='*40)

    # Create data loaders
    train_dataset = AQIDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMPredictor(input_size, hidden_size=hidden_size, num_layers=2, dropout=0.2)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val).to(device))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_val).to(device)).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        val_pred = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model, device, 'lstm', {'hidden_size': hidden_size, 'seq_length': X_train.shape[1]}


def train_lstm_model(feature_cols: list = None, n_features: int = 30):
    """Main LSTM training function."""
    if not HAS_TORCH:
        print("PyTorch not available, skipping LSTM model")
        return None, None, None

    print("="*60)
    print("LSTM Model Training")
    print("="*60)

    # Load data
    df = load_featured_data()

    # Get feature columns
    if feature_cols is None:
        target_cols = [col for col in df.columns if 'target' in col]
        feature_cols = [col for col in df.columns
                       if col not in target_cols and col not in ['aqi', 'aqi_category']]

    # Prepare data
    seq_length = 72  # 3 days of history
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        df, feature_cols, seq_length=seq_length
    )

    # Train LSTM
    model, device, name, params = train_lstm(
        X_train, y_train, X_val, y_val,
        input_size=n_features,
        epochs=100,
        batch_size=64,
        hidden_size=64,
        lr=0.001
    )

    if model is not None:
        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'seq_length': seq_length,
            'feature_cols': feature_cols,
            'device': str(device)
        }

        filepath = os.path.join(MODELS_DIR, 'lstm.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nSaved LSTM to {filepath}")

    return model, (X_train, X_val, X_test, y_train, y_val, y_test), scaler


def predict_lstm(model_data, X_test):
    """Make predictions with LSTM model."""
    model = model_data['model']
    scaler = model_data['scaler']
    seq_length = model_data['seq_length']
    device = torch.device(model_data['device'])

    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return pred


if __name__ == "__main__":
    # Test LSTM training
    model, data, scaler = train_lstm_model()
