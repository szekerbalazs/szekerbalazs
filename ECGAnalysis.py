import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

from scipy.signal import medfilt
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm


def preprocess_ecg_signals(
    ecg_signals,
    wavelet="db4",
    decomposition_level=6,
    median_filter_kernel=9,
    min_length=8000,
):
    """
    Preprocess ECG signals using wavelet transform, median filtering, and length adjustment.

    Parameters:
        ecg_signals (list or np.ndarray): List or array of raw ECG signals.
        wavelet (str): Wavelet type to use for wavelet transform. Default is 'db4'.
        decomposition_level (int): The level of decomposition for wavelet transform. Default is 6.
        median_filter_kernel (int): Kernel size for median filtering. Default is 9.
        min_length (int): The minimum required signal length. Signals shorter than this will be looped.

    Returns:
        np.ndarray: Preprocessed ECG signals of consistent length.
    """
    preprocessed_signals = []

    for signal in ecg_signals:
        # Remove trailing NaNs
        signal = signal[~np.isnan(signal)]

        # Apply wavelet transform for denoising
        coeffs = pywt.wavedec(signal, wavelet, level=decomposition_level)

        # Apply soft thresholding to coefficients for noise reduction
        threshold = np.sqrt(2 * np.log(len(signal)))  # Universal threshold
        denoised_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]

        # Reconstruct the signal
        wavelet_filtered_signal = pywt.waverec(denoised_coeffs, wavelet)

        # Apply median filtering to eliminate baseline drift
        filtered_signal = medfilt(
            wavelet_filtered_signal, kernel_size=median_filter_kernel
        )

        # Adjust the signal length
        if len(filtered_signal) < min_length:
            # Loop the signal to meet the minimum length
            num_repeats = (min_length + len(filtered_signal) - 1) // len(
                filtered_signal
            )
            filtered_signal = np.tile(filtered_signal, num_repeats)[:min_length]
        else:
            # Trim the signal to the minimum length
            filtered_signal = filtered_signal[:min_length]

        # Add the processed signal to the list
        preprocessed_signals.append(filtered_signal)

    return np.array(preprocessed_signals)

class ECGClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4, kernel_size=3):
        super(ECGClassifier, self).__init__()

        # First Conv1D block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second Conv1D block
        self.conv5 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1)
        self.bn8 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third Conv1D block
        self.conv9 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=1)
        self.bn9 = nn.BatchNorm1d(128)
        self.conv10 = nn.Conv1d(128, 128, kernel_size=kernel_size, stride=1)
        self.bn10 = nn.BatchNorm1d(128)
        self.conv11 = nn.Conv1d(128, 128, kernel_size=kernel_size, stride=1)
        self.bn11 = nn.BatchNorm1d(128)
        self.conv12 = nn.Conv1d(128, 128, kernel_size=kernel_size, stride=1)
        self.bn12 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=3)
        self.dropout3 = nn.Dropout(dropout_rate)

        # BiLSTM layers
        self.bilstm1 = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Convolutional block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolutional block 2
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Convolutional block 3
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Prepare input for LSTM (batch, channels, sequence_length -> batch, sequence_length, channels)
        x = x.permute(0, 2, 1)

        # BiLSTM layers
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        # Pooling (optional: can use global average or max pooling for dimensionality reduction)
        x = torch.mean(x, dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


# MSTE Loss Function (Mean Squared Tangent Error)
class MSTE_Loss(nn.Module):
    def __init__(self):
        super(MSTE_Loss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true_one_hot = F.one_hot(y_pred, num_classes=4).float()
        return torch.mean(torch.tan(torch.pow(y_true_one_hot - y_true, 2)))


# Load datasets
train = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
train_y = train["y"].to_numpy()

# Extract ECG signals
train_signals = train.drop(columns=["y"]).values  # Assuming rows are samples and columns are time points
test_signals = test.values

# Preprocess signals
train_signals_preprocessed = preprocess_ecg_signals(train_signals)
test_signals_preprocessed = preprocess_ecg_signals(test_signals)

# Convert to torch tensors
X_train_full = torch.tensor(train_signals_preprocessed, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(test_signals_preprocessed, dtype=torch.float32).unsqueeze(1)
y_train_full = torch.tensor(train_y, dtype=torch.long)

# Split data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    stratify=y_train_full,
    random_state=42,
)

# Convert to DataLoader for batching
train_data = TensorDataset(X_train_split, y_train_split)
val_data = TensorDataset(X_val_split, y_val_split)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Instantiate model, loss function, optimizer, and scheduler
model = ECGClassifier().cuda()
criterion = MSTE_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

best_val_f1 = 0.0
best_epoch = 0
best_model_wts = None
max_epochs = 50

# Training to find the best epoch
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # Calculate validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_f1 = f1_score(all_labels, all_preds, average="micro")
    scheduler.step()

    print(
        f"Epoch {epoch+1}/{max_epochs}, Training Loss: {avg_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}"
    )

    # Track the best epoch
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        best_model_wts = model.state_dict()
        print(f"New best F1 Score: {best_val_f1:.4f} at epoch {best_epoch}")

# Retrain on the full dataset for the best number of epochs
print(f"Retraining on the full dataset for {best_epoch} epochs...")
full_data = TensorDataset(X_train_full, y_train_full)
full_loader = DataLoader(full_data, batch_size=32, shuffle=True)

model = ECGClassifier().cuda()
model.load_state_dict(best_model_wts)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

for epoch in range(best_epoch + 5):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(full_loader, desc=f"Epoch {epoch+1}/{best_epoch}"):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(full_loader)
    scheduler.step()
    print(f"Epoch {epoch+1}/{best_epoch}, Training Loss: {avg_loss:.4f}")

# Evaluate on the test set
test_loader = DataLoader(X_test, batch_size=16, shuffle=False)

model.eval()
all_preds = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

# Save predictions
submission = pd.DataFrame({"id": range(len(all_preds)), "y": all_preds})
submission.to_csv("test_predictions.csv", index=False)

print("Predictions saved to test_predictions.csv")


# 9000, 0.4 dropout, base, 0.773
# 9000, 0.4 dropout, no padding, 0.801
