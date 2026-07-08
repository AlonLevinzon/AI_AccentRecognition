import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class AccentMultiModalCNN(nn.Module):
    # Reduced dropout to 0.35
    def __init__(self, num_classes, phonetic_vector_size=55, dropout_rate=0.35, input_shape=(64, 63)):
        super(AccentMultiModalCNN, self).__init__()
        
        self.input_norm = nn.BatchNorm2d(1)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        self.cnn_flatten_dim = self._get_flatten_dim(input_shape)
        
        combined_dim = self.cnn_flatten_dim + phonetic_vector_size
        self.fc1 = nn.Linear(combined_dim, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_flatten_dim(self, input_shape):
        h, w = input_shape
        h = (h + 2*1 - 3) // 2 + 1
        w = (w + 2*1 - 3) // 2 + 1
        h = (h + 2*1 - 3) // 2 + 1
        w = (w + 2*1 - 3) // 2 + 1
        return 64 * h * w

    def forward(self, x_spec, x_phonetic):
        x_spec = self.input_norm(x_spec)
        
        x_cnn = self.conv1(x_spec)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.pool1(x_cnn)
        
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.bn2(x_cnn)
        x_cnn = self.pool2(x_cnn)
        
        x_cnn_flat = x_cnn.view(x_cnn.size(0), -1)
        x_combined = torch.cat((x_cnn_flat, x_phonetic), dim=1)
        
        x_out = self.fc1(x_combined)
        x_out = self.relu(x_out)
        x_out = self.dropout(x_out)
        
        x_logits = self.fc2(x_out)
        probas = self.softmax(x_logits)
        
        return x_logits, probas


class SpectrogramNumpyDataset(Dataset):
    def __init__(self, X_phonetic, y_labels, filenames, npy_dir, is_training=False):
        phonetic_np = X_phonetic.values.astype(np.float32)
        labels_np = np.array(y_labels, dtype=np.int64)
        
        self.phonetic_features = torch.from_numpy(phonetic_np)
        self.labels = torch.from_numpy(labels_np)
        self.filenames = filenames
        self.npy_dir = npy_dir
        self.is_training = is_training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = str(self.filenames[idx])
        base_name = os.path.splitext(filename)[0] + '.npy'
        npy_path = os.path.join(self.npy_dir, base_name)

        if os.path.exists(npy_path):
            mel_spec = np.load(npy_path)
            x_spec = torch.from_numpy(mel_spec).float().unsqueeze(0)
            # Removed destructive SpecAugment masking for small datasets
        else:
            x_spec = torch.zeros(1, 64, 63, dtype=torch.float32)

        x_phonetic = self.phonetic_features[idx]
        label = self.labels[idx]

        return x_spec, x_phonetic, label


def prepare_pytorch_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    base_dir = "official_audio_mixed"
    npy_base_dir = "official_cnn_spectrograms"

    train_filenames = pd.read_csv(f"{base_dir}/train_features.csv")['filename'].tolist()
    val_filenames = pd.read_csv(f"{base_dir}/validation_features.csv")['filename'].tolist()
    test_filenames = pd.read_csv(f"{base_dir}/test_features.csv")['filename'].tolist()

    # IMPROVEMENT: Only the training dataset gets the augmentation enabled
    train_dataset = SpectrogramNumpyDataset(X_train, y_train, train_filenames, os.path.join(npy_base_dir, "train"), is_training=True)
    val_dataset = SpectrogramNumpyDataset(X_val, y_val, val_filenames, os.path.join(npy_base_dir, "validation"), is_training=False)
    test_dataset = SpectrogramNumpyDataset(X_test, y_test, test_filenames, os.path.join(npy_base_dir, "test"), is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x_spec, x_phonetic, labels in dataloader:
        x_spec = x_spec.to(device)
        x_phonetic = x_phonetic.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(x_spec, x_phonetic)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x_spec.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, correct / total


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_spec, x_phonetic, labels in dataloader:
            x_spec = x_spec.to(device)
            x_phonetic = x_phonetic.to(device)
            labels = labels.to(device)
            
            logits, _ = model(x_spec, x_phonetic)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * x_spec.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total