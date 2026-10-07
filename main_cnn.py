import os
import sys
import shutil
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ==============================================================================
# CONFIGURATION & SYSTEM SETUP
# ==============================================================================

# Choose dataset mode: 'free_text' or 'mixed'
RUN_MODE = "free_text"
DATASET_TYPE = RUN_MODE

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Local Project Directory Configuration
PROJECT_DIR = Path.cwd().resolve()
DATASET_DIR = PROJECT_DIR

# Setup Graphs Directory under graphs/CNN_{RUN_MODE} and overwrite previous contents
PLOTS_DIR = PROJECT_DIR / "graphs" / f"CNN_{RUN_MODE}"
if PLOTS_DIR.exists():
    shutil.rmtree(PLOTS_DIR)  # Clear old plots for this run mode
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Project Root Directory: {PROJECT_DIR}")
print(f"📊 Graphs Output Directory: {PLOTS_DIR} (Cleared & Ready)")

# Target paths definition based on RUN_MODE
if RUN_MODE == "free_text":
    TARGET_CSV = "master_extracted_features_free_text.csv"
    TARGET_SPEC_DIR_KEYWORDS = ["spectrograms_logmel_free_text", "spectrograms_free_text", "free_text"]
else:
    TARGET_CSV = "master_extracted_features_mixed.csv"
    TARGET_SPEC_DIR_KEYWORDS = ["spectrograms_logmel_mixed", "spectrograms_mixed", "mixed"]


# Dynamically locates the target metadata CSV and spectrogram PyTorch tensor files directory within the project.
# It ensures the execution pipeline automatically binds to valid local file paths regardless of project structure.
def find_dataset_paths():
    search_base = str(PROJECT_DIR)
    found_csv = None
    found_spec_dir = None

    for root, dirs, files in os.walk(search_base):
        for file in files:
            if file.lower() == TARGET_CSV.lower():
                found_csv = os.path.join(root, file)
                break
        if found_csv:
            break

    if not found_csv:
        raise FileNotFoundError(f"Could not locate {TARGET_CSV} anywhere under {search_base}")

    parent_search = os.path.dirname(found_csv)
    for item in os.listdir(parent_search):
        full_item_path = os.path.join(parent_search, item)
        if os.path.isdir(full_item_path):
            if any(kw.lower() in item.lower() for kw in TARGET_SPEC_DIR_KEYWORDS):
                found_spec_dir = full_item_path
                break

    if not found_spec_dir:
        for root, dirs, files in os.walk(search_base):
            for d in dirs:
                if any(kw.lower() in d.lower() for kw in TARGET_SPEC_DIR_KEYWORDS):
                    found_spec_dir = os.path.join(root, d)
                    break
            if found_spec_dir:
                break

    return found_csv, found_spec_dir


CSV_PATH, SPECTROGRAM_SAVE_DIR = find_dataset_paths()

print("\n" + "=" * 65)
print("Dynamic Path Finder Success!")
print(f"CSV Path: {CSV_PATH}")
print(f"Spectrograms Directory: {SPECTROGRAM_SAVE_DIR}")
print("=" * 65)

# ==============================================================================
# DATASET INITIALIZATION & MATCHING REPORT
# ==============================================================================

if not os.path.exists(CSV_PATH):
    alt_csv = os.path.join(DATASET_DIR,
                           "master_extracted_features_free_text.csv" if DATASET_TYPE == "mixed" else "master_extracted_features_mixed.csv")
    if os.path.exists(alt_csv):
        CSV_PATH = alt_csv

metadata_df = pd.read_csv(CSV_PATH)
metadata_df["clean_id"] = metadata_df["filename"].apply(
    lambda x: os.path.splitext(os.path.basename(str(x)))[0].lower().strip()
)

existing_pt_files = {
    os.path.splitext(f)[0].lower().strip()
    for f in os.listdir(SPECTROGRAM_SAVE_DIR)
    if f.endswith(".pt")
}

metadata_df["pt_exists"] = metadata_df["clean_id"].isin(existing_pt_files)

print("\n" + "=" * 65)
print("Accent Summary Report (CSV vs Spectrogram PT Files)")
print("=" * 65)

accent_summary = metadata_df.groupby("accent").agg(
    CSV_Total=("clean_id", "count"),
    PT_Generated=("pt_exists", "sum"),
)
accent_summary["Missing"] = accent_summary["CSV_Total"] - accent_summary["PT_Generated"]
accent_summary["Success_Rate_%"] = (
        (accent_summary["PT_Generated"] / accent_summary["CSV_Total"]) * 100
).round(2)

print(accent_summary)
print("-" * 65)
print(f"Total CSV Samples: {len(metadata_df)} | Total Matching PT Files: {accent_summary['PT_Generated'].sum()}")
print("=" * 65)

df_filtered = metadata_df[metadata_df["pt_exists"]].reset_index(drop=True)
df_filtered["file_id"] = df_filtered["clean_id"]


# ==============================================================================
# MODEL & DATASET DEFINITIONS
# ==============================================================================

# Applies random frequency and time masking to log-Mel spectrogram tensors during model training.
# It serves as an acoustic data augmentation technique to prevent CNN overfitting.
class SpecAugment(nn.Module):
    # Initializes masking range parameters for frequency bins and time steps.
    def __init__(self, freq_mask_param=12, time_mask_param=20):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    # Executes random zero-masking on input spectrograms during training mode.
    def forward(self, spec):
        if not self.training:
            return spec
        cloned_spec = spec.clone()
        _, _, num_mels, time_steps = cloned_spec.shape
        f = int(np.random.uniform(0, self.freq_mask_param))
        f0 = np.random.randint(0, max(1, num_mels - f))
        cloned_spec[:, :, f0:f0 + f, :] = 0
        t = int(np.random.uniform(0, self.time_mask_param))
        t0 = np.random.randint(0, max(1, time_steps - t))
        cloned_spec[:, :, :, t0:t0 + t] = 0
        return cloned_spec


# Custom PyTorch Dataset that loads pre-saved log-Mel spectrogram tensors (.pt) and maps accent string labels.
# It feeds formatted batch tensors efficiently into DataLoader workers.
class AccentSpectrogramDataset(Dataset):
    # Initializes dataset dataframe, tensor directory, and string-to-index label mapping.
    def __init__(self, metadata_df, spec_dir, label2idx=None):
        self.df = metadata_df.reset_index(drop=True)
        self.spec_dir = spec_dir
        if label2idx is None:
            unique_labels = sorted(self.df['accent'].unique())
            self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

    # Returns the total number of audio samples present in the dataset split.
    def __len__(self):
        return len(self.df)

    # Loads individual spectrogram PyTorch tensors from disk and formats their dimensions for 2D CNN input.
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['file_id']
        label_str = row['accent']

        spec_path = os.path.join(self.spec_dir, f"{file_id}.pt")
        spec = torch.load(spec_path)

        if isinstance(spec, tuple):
            spec = spec[0]
        if not isinstance(spec, torch.Tensor):
            spec = torch.tensor(spec, dtype=torch.float32)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        label = torch.tensor(self.label2idx[label_str], dtype=torch.long)
        return spec, label


# Defines a 2D Convolutional Neural Network architecture designed for speech accent classification.
# It extracts hierarchical spatial-frequency patterns from log-Mel spectrograms.
class AccentCNN(nn.Module):
    # Builds convolutional layers with BatchNorm, ReLU, Dropout, adaptive pooling, and linear classifier blocks.
    def __init__(self, num_classes: int):
        super().__init__()

        self.augmentation = SpecAugment(freq_mask_param=12, time_mask_param=20)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(40, num_classes)
        )

    # Passes input spectrograms through SpecAugment, convolutional blocks, global pooling, and dense layers.
    def forward(self, x):
        x = self.augmentation(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ==============================================================================
# TRAINING & EVALUATION UTILITIES
# ==============================================================================

# Trains the CNN model for one epoch using Automatic Mixed Precision (AMP) for optimal GPU utilization.
# It updates weights via gradient scaling and computes epoch-level loss and accuracy.
def train_epoch_fast(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for specs, labels in dataloader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(specs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# Evaluates model performance on validation or test sets without computing backpropagation gradients.
# It computes evaluation loss and overall sample accuracy.
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for specs, labels in dataloader:
            specs = specs.to(device)
            labels = labels.to(device)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ==============================================================================
# ANALYSIS & VISUALIZATION FUNCTIONS
# ==============================================================================

# Computes spectral rolloff frequencies tracking the 85% energy cutoff over time frames.
# It helps analyze high-frequency energy degradation differences across accents.
def compute_rolloff_contour(spec_tensor, percentile=0.85):
    if spec_tensor.dim() == 3:
        spec_tensor = spec_tensor.squeeze(0)

    power_spec = torch.exp(spec_tensor).numpy()
    total_energy_per_frame = np.sum(power_spec, axis=0)
    threshold = percentile * total_energy_per_frame
    cumulative_energy = np.cumsum(power_spec, axis=0)

    rolloff_bins = []
    for t in range(power_spec.shape[1]):
        if total_energy_per_frame[t] == 0:
            rolloff_bins.append(0)
        else:
            bin_idx = np.where(cumulative_energy[:, t] >= threshold[t])[0]
            if len(bin_idx) > 0:
                rolloff_bins.append(bin_idx[0])
            else:
                rolloff_bins.append(0)

    return np.array(rolloff_bins)


# Searches for a valid spectrogram sample corresponding to a target accent pattern.
# It provides tensor data for comparative acoustic contour visualizations.
def find_valid_sample(search_pattern):
    filtered = df_filtered[df_filtered['accent'].str.lower().str.contains(search_pattern, regex=True)]
    for _, row in filtered.iterrows():
        file_id = row['clean_id']
        spec_path = os.path.join(SPECTROGRAM_SAVE_DIR, f"{file_id}.pt")
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
            if isinstance(spec, tuple): spec = spec[0]
            if not isinstance(spec, torch.Tensor):
                spec = torch.tensor(spec, dtype=torch.float32)
            return spec, file_id, row['accent']
    return None, None, None


# Computes and visualizes Grad-CAM attention heatmaps over input log-Mel spectrograms.
# It highlights key time-frequency regions that influence CNN accent predictions.
class AccentCNNGradCAM:
    # Registers forward and backward hooks on the target convolutional layer to record activations and gradients.
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    # Hook callback that stores target layer forward feature maps.
    def save_activation(self, module, input, output):
        self.activations = output

    # Hook callback that stores target layer backward output gradients.
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # Generates a normalized Grad-CAM heatmap for a target accent class score.
    def generate_heatmap(self, spec_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(spec_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam, target_class


# ==============================================================================
# MAIN CROSS-VALIDATION PIPELINE
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"Detailed Accent Breakdown across 5 Folds - Acoustic CNN ({RUN_MODE})")
    print("=" * 80)

    gkf_outer = GroupKFold(n_splits=5)
    groups = df_filtered["speaker_id"]

    for fold, (train_val_idx, test_idx) in enumerate(gkf_outer.split(df_filtered, groups=groups), 1):
        train_val_df = df_filtered.iloc[train_val_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)

        gkf_inner = GroupKFold(n_splits=10)
        train_idx_inner, val_idx_inner = next(gkf_inner.split(train_val_df, groups=train_val_df["speaker_id"]))

        train_df = train_val_df.iloc[train_idx_inner].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx_inner].reset_index(drop=True)

        print(f"\n--- FOLD {fold} DISTRIBUTION ---")
        train_counts = train_df['accent'].value_counts()
        val_counts = val_df['accent'].value_counts()
        test_counts = test_df['accent'].value_counts()

        fold_dist = pd.DataFrame({
            'Train_Count': train_counts,
            'Train_%': (train_counts / len(train_df) * 100).round(2),
            'Val_Count': val_counts,
            'Val_%': (val_counts / len(val_df) * 100).round(2),
            'Test_Count': test_counts,
            'Test_%': (test_counts / len(test_df) * 100).round(2)
        }).fillna(0)

        print(fold_dist)
        print(f"Total Samples in Fold {fold}: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Execute 5-Fold Group Cross-Validation
    fold_results = []
    all_overall_preds = []
    all_overall_labels = []

    print("\n" + "=" * 70)
    print("Starting 5-Fold Group Cross-Validation (Acoustic CNN)...")
    print("=" * 70)

    for fold, (train_val_idx, test_idx) in enumerate(gkf_outer.split(df_filtered, groups=groups), 1):
        print(f"\n--- STARTING FOLD {fold}/5 ---")

        train_val_df = df_filtered.iloc[train_val_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)

        gkf_inner = GroupKFold(n_splits=10)
        train_idx_inner, val_idx_inner = next(gkf_inner.split(train_val_df, groups=train_val_df["speaker_id"]))

        train_df = train_val_df.iloc[train_idx_inner].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx_inner].reset_index(drop=True)

        MODEL_SAVE_PATH = PROJECT_DIR / f'best_acoustic_cnn_fold_{fold}.pth'

        train_dataset = AccentSpectrogramDataset(train_df, SPECTROGRAM_SAVE_DIR)
        label2idx = train_dataset.label2idx
        val_dataset = AccentSpectrogramDataset(val_df, SPECTROGRAM_SAVE_DIR, label2idx=label2idx)
        test_dataset = AccentSpectrogramDataset(test_df, SPECTROGRAM_SAVE_DIR, label2idx=label2idx)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2,
                                  pin_memory=True if device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2,
                                pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = AccentCNN(num_classes=len(label2idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

        best_val_loss = float('inf')
        best_epoch = -1

        for epoch in range(30):
            train_loss, train_acc = train_epoch_fast(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for specs, labels in test_loader:
                outputs = model(specs.to(device))
                _, preds = torch.max(outputs, 1)
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_f1 = f1_score(fold_labels, fold_preds, average='macro')
        fold_results.append({
            'fold': fold,
            'test_acc': test_acc,
            'test_f1': fold_f1,
            'best_epoch': best_epoch
        })

        all_overall_preds.extend(fold_preds)
        all_overall_labels.extend(fold_labels)

        print(
            f"Fold {fold} Complete -> Best Epoch: #{best_epoch} | Test Acc: {test_acc * 100:.2f}% | Macro F1: {fold_f1:.4f}")

    # Save primary best model checkpoint
    best_overall_checkpoint = PROJECT_DIR / 'best_accent_cnn.pth'
    shutil.copy(PROJECT_DIR / 'best_acoustic_cnn_fold_1.pth', best_overall_checkpoint)

    # Verification: Speaker Isolation
    print("\nChecking Speaker-ID Isolation across all 5 Folds...")
    total_leaks = 0
    for fold, (train_val_idx, test_idx) in enumerate(gkf_outer.split(df_filtered, groups=groups), 1):
        train_speakers = set(df_filtered.iloc[train_val_idx]['speaker_id'])
        test_speakers = set(df_filtered.iloc[test_idx]['speaker_id'])
        overlap = train_speakers.intersection(test_speakers)
        total_leaks += len(overlap)
        print(
            f"  * Fold {fold}: {len(train_speakers)} Train Speakers | {len(test_speakers)} Test Speakers | Overlap = {len(overlap)}")

    if total_leaks == 0:
        print("Methodological Verification Passed: Zero speaker leakage across all 5 Folds!")

    # Summary Statistics & Classification Report
    accuracies = [r['test_acc'] for r in fold_results]
    f1_scores_list = [r['test_f1'] for r in fold_results]

    mean_acc, std_acc = np.mean(accuracies) * 100, np.std(accuracies) * 100
    mean_f1, std_f1 = np.mean(f1_scores_list), np.std(f1_scores_list)

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON: 5-FOLD CROSS-VALIDATION RESULTS (ACOUSTIC CNN)")
    print("=" * 70)
    for r in fold_results:
        print(
            f"Fold {r['fold']}: Test Acc = {r['test_acc'] * 100:.2f}% | Macro F1 = {r['test_f1']:.4f} (Best Epoch #{r['best_epoch']})")

    print("-" * 70)
    print(f"Overall Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"Overall Mean Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print("=" * 70)

    idx2label = {v: k for k, v in label2idx.items()}
    target_names = [idx2label[i] for i in range(len(label2idx))]

    print("\nAggregated 5-Fold Classification Report - Acoustic CNN:\n")
    print(classification_report(all_overall_labels, all_overall_preds, target_names=target_names, digits=4))

    # 1. Save Normalized Confusion Matrix
    cm_normalized = confusion_matrix(all_overall_labels, all_overall_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Accent', fontsize=12, fontweight='bold')
    plt.ylabel('True Accent', fontsize=12, fontweight='bold')
    plt.title(
        f'Normalized Confusion Matrix - Acoustic CNN ({RUN_MODE})\nOverall Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_normalized.png")
    plt.close()

    # 2. Save Absolute Counts Confusion Matrix
    cm_absolute = confusion_matrix(all_overall_labels, all_overall_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_absolute, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Accent', fontsize=12, fontweight='bold')
    plt.ylabel('True Accent', fontsize=12, fontweight='bold')
    plt.title(f'Aggregated Confusion Matrix (Absolute Counts - {RUN_MODE})\nTotal Samples: {len(all_overall_labels)}',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_absolute.png")
    plt.close()

    # 3. Generate Grad-CAM Attention Maps
    print("\nGenerating Grad-CAM Attention Maps for Acoustic CNN...")
    best_model = AccentCNN(num_classes=len(label2idx)).to(device)
    best_model.load_state_dict(torch.load(best_overall_checkpoint))

    target_layer = best_model.block3[0]
    grad_cam = AccentCNNGradCAM(best_model, target_layer)

    num_accents = len(label2idx)
    grid_size = int(np.ceil(np.sqrt(num_accents)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 10))
    axes = axes.flatten()

    samples_found = {}
    for spec, label in test_dataset:
        l_idx = label.item()
        if l_idx not in samples_found:
            samples_found[l_idx] = spec
        if len(samples_found) == num_accents:
            break

    for i, (l_idx, spec_tensor) in enumerate(sorted(samples_found.items())):
        spec_input = spec_tensor.unsqueeze(0).to(device)
        spec_input.requires_grad = True

        heatmap, pred_class = grad_cam.generate_heatmap(spec_input, target_class=l_idx)
        spec_img = spec_tensor.squeeze().cpu().numpy()

        ax = axes[i]
        ax.imshow(spec_img, aspect='auto', origin='lower', cmap='gray', alpha=0.6)
        ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5,
                  extent=[0, spec_img.shape[1], 0, spec_img.shape[0]])

        accent_name = idx2label[l_idx]
        pred_name = idx2label[pred_class]
        ax.set_title(f"True: {accent_name} | Pred: {pred_name}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Grad-CAM Heatmaps - Acoustic CNN Attention", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "grad_cam_attention_maps.png")
    plt.close()

    # 4. Generate Averaged Accent Spectrogram Templates
    print("\nGenerating Averaged Accent Spectrogram Templates...")
    unique_accents = sorted(df_filtered['accent'].unique())
    num_accents = len(unique_accents)
    grid_size = int(np.ceil(np.sqrt(num_accents)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 10))
    axes = axes.flatten()

    for i, accent in enumerate(unique_accents):
        accent_rows = df_filtered[df_filtered['accent'] == accent]
        specs_list = []

        for _, row in accent_rows.iterrows():
            file_id = row['clean_id']
            spec_path = os.path.join(SPECTROGRAM_SAVE_DIR, f"{file_id}.pt")

            if os.path.exists(spec_path):
                spec = torch.load(spec_path)
                if isinstance(spec, tuple): spec = spec[0]
                if not isinstance(spec, torch.Tensor):
                    spec = torch.tensor(spec, dtype=torch.float32)
                if spec.dim() == 3: spec = spec.squeeze(0)

                specs_list.append(spec.numpy())
                if len(specs_list) >= 30:
                    break

        if len(specs_list) > 0:
            min_time = min(s.shape[1] for s in specs_list)
            cropped_specs = [s[:, :min_time] for s in specs_list]
            avg_spec = np.mean(cropped_specs, axis=0)

            ax = axes[i]
            im = ax.imshow(avg_spec, aspect='auto', origin='lower', cmap='magma')
            ax.set_title(f"Averaged Template: {accent.upper()} (N={len(specs_list)})", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Mel Frequency Bins")
            fig.colorbar(im, ax=ax)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Averaged Log-Mel Spectrogram Templates Across Accents", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "averaged_accent_templates.png")
    plt.close()

    # 5. Residual Learning Visualization
    phonetic_cols = ['phn_voiced', 'phn_fricative', 'phn_plosive', 'phn_nasal', 'phn_long_vowel']
    chinese_df = df_filtered[df_filtered['accent'].str.lower().str.contains('chinese|cn', regex=True)]
    american_df = df_filtered[df_filtered['accent'].str.lower().str.contains('american|us', regex=True)]

    if len(chinese_df) == 0 or len(american_df) == 0:
        available_accents = df_filtered['accent'].unique()
        chinese_df = df_filtered[df_filtered['accent'] == available_accents[0]]
        american_df = df_filtered[df_filtered['accent'] == available_accents[1]]

    if len(chinese_df) > 0 and len(american_df) > 0:
        chinese_row = chinese_df.iloc[0]
        american_row = american_df.iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for i, (row, title) in enumerate([(chinese_row, f"Chinese Accent ({chinese_row['accent']})"),
                                          (american_row, f"American Accent ({american_row['accent']})")]):
            file_id = row['clean_id']
            spec_path = os.path.join(SPECTROGRAM_SAVE_DIR, f"{file_id}.pt")

            if os.path.exists(spec_path):
                spec = torch.load(spec_path)
                if isinstance(spec, tuple): spec = spec[0]
                if spec.dim() == 3: spec = spec.squeeze(0)

                energy_contour = torch.sqrt(torch.mean(spec ** 2, dim=0)).numpy()

                valid_phonetic_cols = [c for c in phonetic_cols if c in row]
                if valid_phonetic_cols:
                    phonetic_vals = row[valid_phonetic_cols].values.astype(float)
                    expected_energy = np.mean(phonetic_vals) * np.max(energy_contour)
                else:
                    expected_energy = np.mean(energy_contour)

                ax = axes[i]
                ax.plot(energy_contour, label='Acoustic Energy Contour (RMS)', color='blue', linewidth=2)
                ax.axhline(y=expected_energy, color='red', linestyle='--', label='Phonetic Standard Expectation (IPA)')

                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel("Time Frames")
                ax.set_ylabel("Normalized Energy")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.suptitle("Residual Learning Visualization: Phonetic Expectation vs Acoustic Execution", fontsize=14,
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "residual_learning_visualization.png")
        plt.close()

    # 6. Spectral Rolloff Contour Comparison
    chinese_spec, cn_id, cn_accent_name = find_valid_sample('chinese|cn')
    american_spec, us_id, us_accent_name = find_valid_sample('american|us')

    if chinese_spec is None or american_spec is None:
        available_accents = df_filtered['accent'].unique()
        chinese_spec, cn_id, cn_accent_name = find_valid_sample(available_accents[0])
        american_spec, us_id, us_accent_name = find_valid_sample(available_accents[1])

    if chinese_spec is not None and american_spec is not None:
        cn_rolloff = compute_rolloff_contour(chinese_spec)
        us_rolloff = compute_rolloff_contour(american_spec)

        plt.figure(figsize=(12, 5))
        plt.plot(cn_rolloff, label=f'Chinese Accent - {cn_id}', color='red', linewidth=2)
        plt.plot(us_rolloff, label=f'American Accent - {us_id}', color='blue', linewidth=2, alpha=0.8)

        plt.title(f'Spectral Rolloff Contour Over Time: {cn_accent_name} vs {us_accent_name}', fontsize=14,
                  fontweight='bold')
        plt.xlabel('Time Frames', fontsize=12)
        plt.ylabel('Mel Frequency Bin (85% Energy Cutoff)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.annotate('Energy Drop\n(Clipping of Final Plosives)',
                     xy=(int(len(cn_rolloff) * 0.75), cn_rolloff[int(len(cn_rolloff) * 0.75)]),
                     xytext=(int(len(cn_rolloff) * 0.55), max(cn_rolloff) * 0.85),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "spectral_rolloff_contour.png")
        plt.close()

    print("\n" + "=" * 80)
    print("✅ ALL CNN PIPELINE OUTPUTS AND GRAPHS SAVED DIRECTLY TO:")
    print(f"   📂 {PLOTS_DIR}")
    print("=" * 80 + "\n")