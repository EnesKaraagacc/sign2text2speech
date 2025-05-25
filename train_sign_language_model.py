"""
train_sign_language_model.py

This script trains a 1D CNN model on Turkish Sign Language features extracted from video
using MediaPipe Holistic. The input data should be sequences of shape (seq_len, 150),
representing 30 time steps of 150 features each (body + hand landmarks).
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# --- Device Configuration ---
print("CUDA available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# --- 1. Load Feature and Label Data ---
save_path = 'processed_data'
X_train = np.load(os.path.join(save_path, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(save_path, 'y_train.npy'), allow_pickle=True)
X_test  = np.load(os.path.join(save_path, 'X_test.npy'),  allow_pickle=True)
y_test  = np.load(os.path.join(save_path, 'y_test.npy'),  allow_pickle=True)

# --- 2. Label Encoding ---
label_encoder = LabelEncoder()
y_all = np.concatenate((y_train, y_test), axis=0)
label_encoder.fit(y_all)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# --- 3. Convert to PyTorch Tensors ---
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# --- 4. Prepare Data Loaders ---
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size)

# --- 5. Define 1D CNN Model ---
class SignLanguageCNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(SignLanguageCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- 6. Train the Model ---
seq_len, feature_dim = X_train.shape[1], X_train.shape[2]
num_classes = len(np.unique(y_train))
model = SignLanguageCNN1D(input_dim=feature_dim, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience = 10
counter = 0
num_epochs = 300

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, train_correct, total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = train_correct / total * 100
    train_loss /= total

    model.eval()
    val_loss, val_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = val_correct / total * 100
    val_loss /= total

    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_sign_language_model_cnn.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# --- 7. Final Test Evaluation ---
model.load_state_dict(torch.load('best_sign_language_model_cnn.pth'))
model.eval()
test_loss, test_correct, total = 0.0, 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test Accuracy: {test_correct / total * 100:.2f}% | Test Loss: {test_loss / total:.4f}")
