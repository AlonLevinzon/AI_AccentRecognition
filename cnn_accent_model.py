import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------
# 1. Hardware Accelerator Setup (GPU / CPU)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------
# 2. SpecAugment (Training Data Augmentation)
# ---------------------------------------------------------
class SpecAugment(nn.Module):
    """Random Frequency and Time Masking for spectrogram augmentation.

    Prevents overfitting by forcing the network to learn robust features.
    """

    def __init__(self, freq_mask_param=15, time_mask_param=25):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, spec):
        # Apply augmentation only during training mode
        if not self.training:
            return spec

        cloned_spec = spec.clone()
        _, _, num_mels, time_steps = cloned_spec.shape

        # Frequency Masking
        f = int(np.random.uniform(0, self.freq_mask_param))
        f0 = np.random.randint(0, max(1, num_mels - f))
        cloned_spec[:, :, f0 : f0 + f, :] = 0

        # Time Masking
        t = int(np.random.uniform(0, self.time_mask_param))
        t0 = np.random.randint(0, max(1, time_steps - t))
        cloned_spec[:, :, :, t0 : t0 + t] = 0

        return cloned_spec


# ---------------------------------------------------------
# 3. Custom PyTorch Dataset
# ---------------------------------------------------------
class AccentSpectrogramDataset(Dataset):
    """Loads preprocessed spectrogram tensors based on metadata CSV file."""

    def __init__(self, metadata_df, spec_dir, label2idx=None):
        self.df = metadata_df.reset_index(drop=True)
        self.spec_dir = spec_dir

        # Build accent string to class index mapping
        if label2idx is None:
            unique_labels = sorted(self.df["accent"].unique())
            self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row["file_id"]  # File identifier
        label_str = row["accent"]

        # Load saved spectrogram tensor (.pt)
        spec_path = os.path.join(self.spec_dir, f"{file_id}.pt")
        spec = torch.load(spec_path)  # Expected shape: [1, n_mels, time_steps]

        label = torch.tensor(self.label2idx[label_str], dtype=torch.long)
        return spec, label


# ---------------------------------------------------------
# 4. CNN Model Architecture (AccentCNN)
# ---------------------------------------------------------
class AccentCNN(nn.Module):
    """Hierarchical 2D CNN architecture with BatchNorm, Dropout, and GAP."""

    def __init__(self, num_classes: int):
        super().__init__()

        # Online SpecAugment layer
        self.augmentation = SpecAugment(freq_mask_param=12, time_mask_param=20)

        # Convolutional Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # Downsamples frequency & time dimensions
            nn.Dropout(0.2),
        )

        # Convolutional Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        # Convolutional Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )

        # Global Average Pooling (GAP)
        # Reduces spatial dimensions (freq & time) to a single mean per feature map
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Classifier (Dense/MLP)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Input tensor shape: [batch_size, 1, n_mels, time_steps]
        x = self.augmentation(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)  # Tensor shape: [batch_size, 128, 1, 1]
        x = torch.flatten(x, 1)  # Tensor shape: [batch_size, 128]

        logits = self.classifier(x)  # Tensor shape: [batch_size, num_classes]
        return logits


# ---------------------------------------------------------
# 5. Training and Evaluation Pipeline
# ---------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Executes one full training epoch over the dataset."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for specs, labels in dataloader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation dataset."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for specs, labels in dataloader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------
# 6. Training Pipeline Function (Fold Integration)
# ---------------------------------------------------------
def run_fold_training(train_df, val_df, spec_dir, num_epochs=30, batch_size=32):
    """Trains the AccentCNN model on a specific GroupKFold split."""
    train_dataset = AccentSpectrogramDataset(train_df, spec_dir)
    val_dataset = AccentSpectrogramDataset(
        val_df, spec_dir, label2idx=train_dataset.label2idx
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.label2idx)
    model = AccentCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}"
        )

        # Model checkpoint based on lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_accent_cnn.pth")

    return model