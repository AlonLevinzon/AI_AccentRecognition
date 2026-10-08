import os

# 1. Force PyTorch to disable MKL & OpenMP threading optimizations causing Windows 0xC0000005 crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

import re
import gc
import glob
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

# Disable internal CPU threading and force deterministic CPU operations
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Disable MKL primitives if available on Windows CPU build
if hasattr(torch.backends, 'mkl'):
    torch.backends.mkl.is_available = lambda: False

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# Global configuration and project directory definitions
PROJECT_DIR = Path.cwd().resolve()
INPUT_DIR = PROJECT_DIR / "archive_cnn_dataset"
PLOTS_DIR = PROJECT_DIR / "graphs" / "archive_cnn"

# Local temporary folders for spectrograms
DIR_LOGMEL = PROJECT_DIR / "temp_spectrograms_logmel"
DIR_LINEAR = PROJECT_DIR / "temp_spectrograms_linear"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Locates dataset CSV or Excel files dynamically within the input folder
def find_metadata_file(search_dir: Path) -> Path:
    found_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith((".csv", ".xlsx", ".xls")):
                found_files.append(Path(root) / file)

    if not found_files:
        raise FileNotFoundError(f"No metadata CSV or Excel files found under {search_dir}")

    return found_files[0]


# Applies dynamic frequency and time zero-masking on input spectrogram tensors during training
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=16, time_mask_param=80):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

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


# In-Memory PyTorch Dataset that loads all tensors into RAM to prevent disk lock crashes
class InMemoryAccentDataset(Dataset):
    def __init__(self, metadata_df, spec_dir, label2idx=None):
        self.df = metadata_df.reset_index(drop=True)
        if label2idx is None:
            unique_labels = sorted(self.df["accent"].unique())
            self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

        self.samples = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            file_id = row["file_id"]
            label_str = row["accent"]

            spec_path = os.path.join(spec_dir, f"{file_id}.pt")
            spec = torch.load(spec_path, map_location='cpu')

            if isinstance(spec, tuple):
                spec = spec[0]
            if not isinstance(spec, torch.Tensor):
                spec = torch.tensor(spec, dtype=torch.float32)
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)

            label = torch.tensor(self.label2idx[label_str], dtype=torch.long)
            self.samples.append((spec, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Implements a 3-Block 2D Convolutional Neural Network (20 -> 40 -> 80 filters) with adaptive pooling
class AccentCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.augmentation = SpecAugment()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.3)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(80, 40), nn.ReLU(), nn.Dropout(0.35), nn.Linear(40, num_classes)
        )

    def forward(self, x):
        x = self.augmentation(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# Loads audio files and extracts standardized 45-second Log-Mel (dB) and Linear Mel spectrograms
def process_and_extract_spectrograms(audio_path: str, target_sr: int = 16000, target_duration_sec: float = 45.0):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    target_length = int(target_sr * target_duration_sec)
    num_samples = waveform.shape[1]

    if num_samples < target_length:
        pad_amount = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > target_length:
        waveform = waveform[:, :target_length]

    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=128, power=2.0
    )
    raw_mel = mel_transform(waveform)

    linear_spec = raw_mel.clone()
    linear_spec = (linear_spec - linear_spec.mean()) / (linear_spec.std() + 1e-6)

    logmel_spec = T.AmplitudeToDB(top_db=80)(raw_mel)
    logmel_spec = (logmel_spec - logmel_spec.mean()) / (logmel_spec.std() + 1e-6)

    return logmel_spec, linear_spec


# Reads CSV and Excel files robustly across multiple encoding formats
def smart_read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    for enc in ['utf-8', 'latin1', 'utf-8-sig', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')


# Maps extracted filename prefixes to four main target accent categories
def map_to_main_accent(filename_str):
    ACCENT_MAP = {
        "english": "usa",
        "mandarin": "chinese",
        "bengali": "indian", "hindi": "indian", "urdu": "indian", "punjabi": "indian",
        "marathi": "indian", "tamil": "indian", "telugu": "indian",
        "hausa": "nigerian", "yoruba": "nigerian", "krio": "nigerian", "twi": "nigerian",
        "fanti": "nigerian", "igbo": "nigerian", "mende": "nigerian", "ibibio": "nigerian",
        "ashanti": "nigerian", "kanuri": "nigerian", "akan": "nigerian", "edo": "nigerian", "ga": "nigerian"
    }
    clean_name = os.path.basename(str(filename_str)).lower().strip()
    match = re.match(r"^([a-zA-Z]+)", clean_name)
    if match:
        prefix = match.group(1)
        return ACCENT_MAP.get(prefix, "other")
    return "other"


# Trains the AccentCNN model on training split using pre-loaded RAM dataset
def train_and_track(spec_dir, train_df, val_df, name="Model", save_filename="best_model.pth", num_epochs=30, batch_size=32):
    train_dataset = InMemoryAccentDataset(train_df, spec_dir)
    val_dataset = InMemoryAccentDataset(val_df, spec_dir, label2idx=train_dataset.label2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AccentCNN(num_classes=len(train_dataset.label2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nStarting Training: {name}")
    print("-" * 50)

    for epoch in range(num_epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            t_correct += (preds == labels).sum().item()
            t_total += labels.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * specs.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        train_acc, val_acc = t_correct / t_total, v_correct / v_total
        val_loss = v_loss / v_total
        scheduler.step(val_loss)

        history['train_loss'].append(t_loss / t_total)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_filename)

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    del model, train_loader, val_loader
    gc.collect()

    return history


# Evaluates trained model checkpoint on unseen test split
def evaluate_test_set(spec_dir, test_df, train_val_df, model_name="Model", weight_path="best_model.pth"):
    unique_labels = sorted(train_val_df['accent'].unique())
    label2idx = {label: i for i, label in enumerate(unique_labels)}

    test_dataset = InMemoryAccentDataset(test_df, spec_dir, label2idx=label2idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = AccentCNN(num_classes=len(label2idx)).to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Successfully loaded checkpoint: {weight_path}")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            outputs = model(specs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    idx2label = {v: k for k, v in label2idx.items()}
    target_names = [idx2label[i] for i in range(len(label2idx))]

    print(f"\nClassification Report - Test Set ({model_name}):\n")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Accent')
    plt.ylabel('True Accent')
    plt.title(f'Confusion Matrix - {model_name} (Test Set)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.close()

    return test_acc


# Main execution entry point
if __name__ == "__main__":
    print(f"Using compute device: {device}")

    # Clean existing graph outputs and temp folders on fresh start
    if PLOTS_DIR.exists():
        shutil.rmtree(PLOTS_DIR)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if DIR_LOGMEL.exists():
        shutil.rmtree(DIR_LOGMEL)
    if DIR_LINEAR.exists():
        shutil.rmtree(DIR_LINEAR)

    DIR_LOGMEL.mkdir(parents=True, exist_ok=True)
    DIR_LINEAR.mkdir(parents=True, exist_ok=True)

    try:
        csv_path = find_metadata_file(INPUT_DIR)
        print(f"Found Metadata File: {csv_path}")

        all_meta_files = glob.glob(os.path.join(INPUT_DIR, "**/*.csv"), recursive=True) + \
                         glob.glob(os.path.join(INPUT_DIR, "**/*.xlsx"), recursive=True)
        if not all_meta_files:
            all_meta_files = [str(csv_path)]

        dfs = [smart_read_file(f) for f in all_meta_files]
        metadata_df = pd.concat(dfs, ignore_index=True)

        filename_col = "split_filename" if "split_filename" in metadata_df.columns else "filename"
        metadata_df["file_id"] = metadata_df[filename_col].apply(
            lambda x: os.path.splitext(os.path.basename(str(x)))[0].lower().strip()
        )

        metadata_df["accent"] = metadata_df["file_id"].apply(map_to_main_accent)
        metadata_df = metadata_df[metadata_df["accent"] != "other"].reset_index(drop=True)

        valid_extensions = {".wav", ".mp3", ".flac", ".ogg"}
        audio_files_map = {}

        for root, _, files in os.walk(INPUT_DIR):
            for f in files:
                if Path(f).suffix.lower() in valid_extensions:
                    file_id = os.path.splitext(f)[0].lower().strip()
                    audio_files_map[file_id] = os.path.join(root, f)

        print("Generating Spectrograms into Local Folder...")
        for file_id, audio_path in tqdm(audio_files_map.items(), desc="Generating Spectrograms"):
            path_log = DIR_LOGMEL / f"{file_id}.pt"
            path_lin = DIR_LINEAR / f"{file_id}.pt"

            if path_log.exists() and path_lin.exists():
                continue

            try:
                log_tensor, lin_tensor = process_and_extract_spectrograms(audio_path)
                torch.save(log_tensor, path_log)
                torch.save(lin_tensor, path_lin)
            except Exception as e:
                print(f"Error processing {file_id}: {e}")

        gc.collect()

        existing_pt_files = {f.stem.lower().strip() for f in DIR_LOGMEL.glob("*.pt")}
        metadata_df = metadata_df[metadata_df["file_id"].isin(existing_pt_files)].reset_index(drop=True)

        print(f"Successfully processed {len(existing_pt_files)} spectrogram tensors.")

        speaker_col = "speakerid" if "speakerid" in metadata_df.columns else "speaker_id"
        if speaker_col not in metadata_df.columns:
            metadata_df[speaker_col] = metadata_df["file_id"]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(gss.split(metadata_df, groups=metadata_df[speaker_col]))

        train_val_df = metadata_df.iloc[train_val_idx].reset_index(drop=True)
        test_df = metadata_df.iloc[test_idx].reset_index(drop=True)

        gkf = GroupKFold(n_splits=5)
        train_val_df["fold"] = -1
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_val_df, groups=train_val_df[speaker_col])):
            train_val_df.loc[val_idx, "fold"] = fold_idx

        train_df = train_val_df[train_val_df["fold"] != 0].reset_index(drop=True)
        val_df = train_val_df[train_val_df["fold"] == 0].reset_index(drop=True)

        print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

        path_best_logmel = PROJECT_DIR / "best_logmel.pth"
        path_best_linear = PROJECT_DIR / "best_linear.pth"

        history_logmel = train_and_track(DIR_LOGMEL, train_df, val_df, name="Log-Mel (Decibel)", save_filename=str(path_best_logmel), num_epochs=30)
        history_linear = train_and_track(DIR_LINEAR, train_df, val_df, name="Linear Mel", save_filename=str(path_best_linear), num_epochs=30)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history_logmel['val_acc'], label='Log-Mel (Decibels)', color='blue', linewidth=2)
        plt.plot(history_linear['val_acc'], label='Linear Mel', color='orange', linewidth=2, linestyle='--')
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history_logmel['val_loss'], label='Log-Mel (Decibels)', color='blue', linewidth=2)
        plt.plot(history_linear['val_loss'], label='Linear Mel', color='orange', linewidth=2, linestyle='--')
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "validation_curves_comparison.png")
        plt.close()

        test_acc_log = evaluate_test_set(DIR_LOGMEL, test_df, train_val_df, model_name="Log-Mel (Decibel)", weight_path=str(path_best_logmel))
        test_acc_lin = evaluate_test_set(DIR_LINEAR, test_df, train_val_df, model_name="Linear Mel", weight_path=str(path_best_linear))

        plt.figure(figsize=(8, 5))
        bars = plt.bar(['Log-Mel', 'Linear Mel'], [test_acc_log * 100, test_acc_lin * 100], color=['blue', 'orange'])
        plt.title('Final Unseen Test Set Accuracy (%)')
        plt.ylabel('Accuracy (%)')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "test_accuracy_comparison.png")
        plt.close()

        print("\n" + "=" * 65)
        print("ALL GRAPH OUTPUTS AND RESULTS SAVED TO:")
        print(f" -> {PLOTS_DIR}")
        print("=" * 65)

    finally:
        gc.collect()
        if DIR_LOGMEL.exists():
            shutil.rmtree(DIR_LOGMEL, ignore_errors=True)
        if DIR_LINEAR.exists():
            shutil.rmtree(DIR_LINEAR, ignore_errors=True)
        print("TEMPORARY SPECTROGRAMS AUTOMATICALLY CLEANED UP FROM DISK!")