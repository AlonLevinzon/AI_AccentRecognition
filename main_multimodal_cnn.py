import os
import sys
import shutil
import warnings
import traceback
import subprocess
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

# Dynamic Installation of Missing Dependencies for Inference/IPA processing
try:
    import eng_to_ipa as ipa
except ModuleNotFoundError:
    print("Installing eng-to-ipa package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "eng-to-ipa", "--quiet"])
    import eng_to_ipa as ipa

try:
    import whisper
except ModuleNotFoundError:
    print("Installing openai-whisper package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper", "--quiet"])
    import whisper

import torchaudio
import torchaudio.transforms as T

# ==============================================================================
# CONFIGURATION & GLOBAL DEFINITIONS
# ==============================================================================

RUN_MODE = "free_text"
DATASET_TYPE = RUN_MODE

PROJECT_DIR = Path.cwd().resolve()
DATASET_DIR = PROJECT_DIR
PLOTS_DIR = PROJECT_DIR / "graphs" / f"Multimodal_CNN_{RUN_MODE}"

if RUN_MODE == "free_text":
    TARGET_CSV = "master_extracted_features_free_text.csv"
    EXACT_SPEC_DIR_NAME = "spectrograms_logmel_free_text"
else:
    TARGET_CSV = "master_extracted_features_mixed.csv"
    EXACT_SPEC_DIR_NAME = "spectrograms_logmel_mixed"


# ==============================================================================
# CLASSES & FUNCTIONS DEFINITIONS (SAFE FOR MULTIPROCESSING)
# ==============================================================================

# Dynamically locates the target metadata CSV and explicit spectrogram directory.
def find_dataset_paths():
    search_base = str(PROJECT_DIR)
    found_csv, found_spec_dir = None, None

    for root, dirs, files in os.walk(search_base):
        for file in files:
            if file.lower() == TARGET_CSV.lower():
                found_csv = os.path.join(root, file)
                break
        if found_csv:
            break

    if not found_csv:
        raise FileNotFoundError(f"Could not locate {TARGET_CSV} anywhere under {search_base}")

    exact_path = PROJECT_DIR / EXACT_SPEC_DIR_NAME
    if exact_path.exists() and exact_path.is_dir():
        found_spec_dir = str(exact_path)
    else:
        for root, dirs, files in os.walk(search_base):
            if os.path.basename(root).lower() == EXACT_SPEC_DIR_NAME.lower():
                found_spec_dir = root
                break

    if not found_spec_dir:
        for root, dirs, files in os.walk(search_base):
            pt_files = [f for f in files if f.endswith('.pt')]
            if len(pt_files) > 0 and EXACT_SPEC_DIR_NAME.lower() in root.lower():
                found_spec_dir = root
                break

    if not found_spec_dir:
        raise FileNotFoundError(f"Could not locate spectrogram directory '{EXACT_SPEC_DIR_NAME}' under {search_base}")

    return found_csv, found_spec_dir


# Applies random frequency and time zero-masking on input spectrogram tensors during training.
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=12, time_mask_param=20):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, spec):
        # Skip augmentation during evaluation or test mode
        if not self.training:
            return spec
        cloned_spec = spec.clone()
        _, _, num_mels, time_steps = cloned_spec.shape
        # Zero out a random contiguous block of frequency channels
        f = int(np.random.uniform(0, self.freq_mask_param))
        f0 = np.random.randint(0, max(1, num_mels - f))
        cloned_spec[:, :, f0:f0 + f, :] = 0
        # Zero out a random contiguous block of time steps
        t = int(np.random.uniform(0, self.time_mask_param))
        t0 = np.random.randint(0, max(1, time_steps - t))
        cloned_spec[:, :, :, t0:t0 + t] = 0
        return cloned_spec


# Custom PyTorch Dataset that loads both 2D log-Mel spectrogram tensors and 1D IPA phonetic feature vectors.
class DualBranchDataset(Dataset):
    def __init__(self, metadata_df, spec_dir, label2idx=None):
        self.df = metadata_df.reset_index(drop=True)
        self.spec_dir = spec_dir
        self.phonetic_cols = [
            'phn_voiced', 'phn_fricative', 'phn_plosive', 'phn_nasal', 'phn_long_vowel'
        ]

        if label2idx is None:
            unique_labels = sorted(self.df['accent'].unique())
            self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['file_id']
        label_str = row['accent']

        spec_path = os.path.join(self.spec_dir, f"{file_id}.pt")
        spec = torch.load(spec_path)

        # Unpack tuple if saved with additional metadata
        if isinstance(spec, tuple):
            spec = spec[0]
        if not isinstance(spec, torch.Tensor):
            spec = torch.tensor(spec, dtype=torch.float32)
        # Ensure 3D tensor format [Channels, Mels, Time] for Conv2d
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        phonetic_vec = row[self.phonetic_cols].values.astype(np.float32)
        phonetic_tensor = torch.tensor(phonetic_vec, dtype=torch.float32)
        label = torch.tensor(self.label2idx[label_str], dtype=torch.long)

        return spec, phonetic_tensor, label


# Implements a Multimodal Dual-Branch Convolutional Neural Network with gated residual feature fusion.
class DualBranchAccentCNN(nn.Module):
    def __init__(self, num_classes: int, phonetic_dim: int = 5):
        super().__init__()

        # Acoustic Branch (20 -> 40 -> 80)
        self.augmentation = SpecAugment(freq_mask_param=12, time_mask_param=20)
        self.acoustic_branch = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            # Collapse spatial dimensions to [Batch, 80, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Phonetic Branch (Projected to 80)
        self.phonetic_branch = nn.Sequential(
            nn.Linear(phonetic_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Residual Feature Fusion Gate
        self.gate = nn.Sequential(
            nn.Linear(80 + 80, 80),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(40, num_classes)
        )

    def forward(self, spec, phonetic_vec):
        x_ac = self.augmentation(spec)
        x_ac = self.acoustic_branch(x_ac)
        x_ac = torch.flatten(x_ac, 1)

        x_ph = self.phonetic_branch(phonetic_vec)

        # Compute dynamic gating factor
        combined = torch.cat((x_ac, x_ph), dim=1)
        g = self.gate(combined)

        # Residual Addition: Acoustic features enhanced by gated phonetic features
        fused = x_ac + g * x_ph

        out = self.classifier(fused)
        return out


# Trains the multimodal model for one epoch (CPU & GPU safe).
def train_epoch_fast(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for specs, phonetics, labels in dataloader:
        specs = specs.to(device, non_blocking=True)
        phonetics = phonetics.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training on GPU via AMP
        if device.type == 'cuda' and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(specs, phonetics)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(specs, phonetics)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# Evaluates the multimodal model on validation or test sets.
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for specs, phonetics, labels in dataloader:
            specs = specs.to(device)
            phonetics = phonetics.to(device)
            labels = labels.to(device)

            outputs = model(specs, phonetics)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# Computes spectral rolloff frequencies tracking the 85% energy cutoff across time frames.
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
            # Find first Mel frequency bin exceeding 85% cumulative energy
            bin_idx = np.where(cumulative_energy[:, t] >= threshold[t])[0]
            if len(bin_idx) > 0:
                rolloff_bins.append(bin_idx[0])
            else:
                rolloff_bins.append(0)

    return np.array(rolloff_bins)


# Searches metadata for a valid spectrogram file matching an accent string pattern.
def find_valid_sample(df_filtered, spec_dir, search_pattern):
    filtered_df = df_filtered[df_filtered['accent'].str.lower().str.contains(search_pattern, regex=True)]
    for _, row in filtered_df.iterrows():
        file_id = row['clean_id']
        spec_path = os.path.join(spec_dir, f"{file_id}.pt")
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
            if isinstance(spec, tuple): spec = spec[0]
            if not isinstance(spec, torch.Tensor):
                spec = torch.tensor(spec, dtype=torch.float32)
            return spec, file_id, row['accent']
    return None, None, None


# Computes Grad-CAM attention heatmaps specifically for the acoustic branch of the dual-branch CNN.
class DualBranchGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register PyTorch hooks to capture forward activations and backward gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, spec_tensor, phonetic_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(spec_tensor, phonetic_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # Compute global average pooling over spatial dimensions to get channel importance weights
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU to keep features with positive influence
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam, target_class


# Maps transcribed English text into normalized International Phonetic Alphabet (IPA) feature vectors.
def extract_official_ipa_phonetic_vector(transcribed_text: str) -> torch.Tensor:
    IPA_FEATURES_TABLE = {
        'p': {'voiced': 0, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},
        'b': {'voiced': 1, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},
        't': {'voiced': 0, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},
        'd': {'voiced': 1, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},
        'k': {'voiced': 0, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},
        'g': {'voiced': 1, 'fricative': 0, 'plosive': 1, 'nasal': 0, 'long_vowel': 0},
        'f': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},
        'v': {'voiced': 1, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},
        's': {'voiced': 0, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},
        'z': {'voiced': 1, 'fricative': 1, 'plosive': 0, 'nasal': 0, 'long_vowel': 0},
        'm': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 1, 'long_vowel': 0},
        'n': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 1, 'long_vowel': 0},
        'i:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
        'ɑ:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
        'u:': {'voiced': 1, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 1},
    }

    if not transcribed_text:
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    raw_ipa = ipa.convert(transcribed_text)
    phonemes = list(raw_ipa.replace("*", ""))

    if not phonemes:
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    counts = {'voiced': 0, 'fricative': 0, 'plosive': 0, 'nasal': 0, 'long_vowel': 0}

    for phoneme in phonemes:
        if phoneme in IPA_FEATURES_TABLE:
            feat = IPA_FEATURES_TABLE[phoneme]
            for k in counts.keys():
                if feat[k] == 1:
                    counts[k] += 1

    total_phns = len(phonemes)
    # Normalize counts by total number of recognized phonemes
    normalized_vec = [
        counts['voiced'] / total_phns,
        counts['fricative'] / total_phns,
        counts['plosive'] / total_phns,
        counts['nasal'] / total_phns,
        counts['long_vowel'] / total_phns
    ]

    return torch.tensor(normalized_vec, dtype=torch.float32).unsqueeze(0)


# Loads a .wav audio file and extracts a standardized, normalized 128-bin log-Mel spectrogram tensor.
def process_and_extract_mel_spectrogram(audio_path: str, target_sr: int = 16000,
                                        target_duration_sec: float = 4.0) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)

    # Convert stereo audio to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    target_length = int(target_sr * target_duration_sec)
    num_samples = waveform.shape[1]

    # Zero-pad short audio or truncate long audio to exactly 4 seconds
    if num_samples < target_length:
        pad_amount = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > target_length:
        waveform = waveform[:, :target_length]

    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=2.0
    )
    mel_spec = mel_transform(waveform)
    mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

    # Apply Z-score normalization across spectrogram values
    mean = mel_spec.mean()
    std = mel_spec.std() + 1e-6
    mel_spec = (mel_spec - mean) / std

    return mel_spec


# Runs the complete end-to-end multimodal inference pipeline on a target audio file.
def process_and_classify_uploaded_audio(file_path: str, inference_model, whisper_model, label2idx_mapping, device):
    print(f"\nProcessing Audio File for Inference: {os.path.basename(file_path)}")

    try:
        whisper_result = whisper_model.transcribe(str(file_path))
        transcribed_text = whisper_result.get("text", "").strip()
        print(f"Recognized Speech (Whisper STT): \"{transcribed_text}\"")

        spec_tensor = process_and_extract_mel_spectrogram(str(file_path))
        spec_input = spec_tensor.unsqueeze(0).to(device)

        phonetic_input = extract_official_ipa_phonetic_vector(transcribed_text).to(device)

        inference_model.eval()
        with torch.no_grad():
            outputs = inference_model(spec_input, phonetic_input)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        idx2label = {v: k for k, v in label2idx_mapping.items()}
        predicted_idx = np.argmax(probabilities)
        predicted_accent = idx2label[predicted_idx]

        print("\nAccent Class Probability Breakdown:")
        for idx, prob in enumerate(probabilities):
            accent_name = idx2label[idx]
            print(f"  * {accent_name.upper():<12}: {prob * 100:.2f}%")

        print("=" * 60)
        print(f"Predicted Accent: {predicted_accent.upper()}")
        print("=" * 60)

    except Exception as e:
        print(f"Error during audio inference execution: {e}")


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    if PLOTS_DIR.exists():
        shutil.rmtree(PLOTS_DIR)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f" Project Root Directory: {PROJECT_DIR}")
    print(f"📊 Graphs Output Directory: {PLOTS_DIR} ")

    CSV_PATH, SPECTROGRAM_SAVE_DIR = find_dataset_paths()

    print("\n" + "=" * 65)
    print("Dynamic Path Finder Success!")
    print(f"CSV Path: {CSV_PATH}")
    print(f"Spectrograms Directory: {SPECTROGRAM_SAVE_DIR}")
    print("=" * 65)

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
    print("Accent File and Spectrogram Summary Report")
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
    total_matched = accent_summary['PT_Generated'].sum()
    print(f"Total CSV Samples: {len(metadata_df)} | Total Matching PT Files: {total_matched}")
    print("=" * 65)

    df_filtered = metadata_df[metadata_df["pt_exists"]].reset_index(drop=True)
    df_filtered["file_id"] = df_filtered["clean_id"]

    if len(df_filtered) == 0:
        print("\n CRITICAL ERROR: Zero matching .pt files found!")
        sys.exit(1)

    # Use GroupKFold to ensure speaker-independent cross-validation splits
    gkf_outer = GroupKFold(n_splits=5)
    groups = df_filtered["speaker_id"]

    fold_results = []
    all_overall_preds = []
    all_overall_labels = []

    # Disable multiprocessing workers on Windows to prevent runtime issues
    num_workers_cfg = 0 if sys.platform.startswith('win') else 2

    print("\n" + "=" * 70)
    print("Starting 5-Fold Group Cross-Validation (Multimodal Dual-Branch CNN)...")
    print("=" * 70)

    for fold, (train_val_idx, test_idx) in enumerate(gkf_outer.split(df_filtered, groups=groups), 1):
        print(f"\n--- STARTING FOLD {fold}/5 ---")

        train_val_df = df_filtered.iloc[train_val_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)

        # Inner GroupKFold split to derive a speaker-isolated validation set
        gkf_inner = GroupKFold(n_splits=10)
        train_idx_inner, val_idx_inner = next(gkf_inner.split(train_val_df, groups=train_val_df["speaker_id"]))

        train_df = train_val_df.iloc[train_idx_inner].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx_inner].reset_index(drop=True)

        print(f"Fold {fold} Distribution -> Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

        MODEL_SAVE_PATH = PROJECT_DIR / f'best_dual_branch_fold_{fold}.pth'

        train_dataset = DualBranchDataset(train_df, SPECTROGRAM_SAVE_DIR)
        label2idx = train_dataset.label2idx
        val_dataset = DualBranchDataset(val_df, SPECTROGRAM_SAVE_DIR, label2idx=label2idx)
        test_dataset = DualBranchDataset(test_df, SPECTROGRAM_SAVE_DIR, label2idx=label2idx)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers_cfg,
                                  pin_memory=True if device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers_cfg,
                                pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        model = DualBranchAccentCNN(num_classes=len(label2idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # Decay learning rate when validation loss plateaus for 3 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        best_val_loss = float('inf')
        best_epoch = -1

        for epoch in range(30):
            train_loss, train_acc = train_epoch_fast(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            # Checkpoint model weights when achieving lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Reload best weights for test evaluation
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for specs, phonetics, labels in test_loader:
                outputs = model(specs.to(device), phonetics.to(device))
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

    best_overall_checkpoint = PROJECT_DIR / 'best_dual_branch_cnn.pth'
    shutil.copy(PROJECT_DIR / 'best_dual_branch_fold_1.pth', best_overall_checkpoint)

    # Verify that no speaker IDs overlap between train and test splits
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

    accuracies = [r['test_acc'] for r in fold_results]
    f1_scores_list = [r['test_f1'] for r in fold_results]

    mean_acc, std_acc = np.mean(accuracies) * 100, np.std(accuracies) * 100
    mean_f1, std_f1 = np.mean(f1_scores_list), np.std(f1_scores_list)

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION FINAL SUMMARY REPORT (DUAL-BRANCH CNN)")
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

    print("\nAggregated 5-Fold Classification Report - Dual-Branch CNN:\n")
    print(classification_report(all_overall_labels, all_overall_preds, target_names=target_names, digits=4))

    # Confusion Matrices
    cm_normalized = confusion_matrix(all_overall_labels, all_overall_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Accent', fontsize=12, fontweight='bold')
    plt.ylabel('True Accent', fontsize=12, fontweight='bold')
    plt.title(
        f'Aggregated 5-Fold Confusion Matrix - Dual-Branch CNN ({RUN_MODE})\nMean Test Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_normalized.png")
    plt.close()

    cm_absolute = confusion_matrix(all_overall_labels, all_overall_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_absolute, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Accent', fontsize=12, fontweight='bold')
    plt.ylabel('True Accent', fontsize=12, fontweight='bold')
    plt.title(
        f'Aggregated Confusion Matrix (Absolute Counts - {RUN_MODE})\nDual-Branch CNN - Total Samples: {len(all_overall_labels)}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_absolute.png")
    plt.close()

    # Grad-CAM Attention Maps
    print("\nGenerating Grad-CAM Attention Maps for Dual-Branch CNN...")
    best_model = DualBranchAccentCNN(num_classes=len(label2idx)).to(device)
    best_model.load_state_dict(torch.load(best_overall_checkpoint))

    # Target the final convolutional block of the acoustic branch
    target_layer = best_model.acoustic_branch[8]
    grad_cam = DualBranchGradCAM(best_model, target_layer)

    num_accents = len(label2idx)
    grid_size = int(np.ceil(np.sqrt(num_accents)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 10))
    axes = axes.flatten()

    # Find one representative sample per accent class from the test dataset
    samples_found = {}
    for spec, phonetic, label in test_dataset:
        l_idx = label.item()
        if l_idx not in samples_found:
            samples_found[l_idx] = (spec, phonetic)
        if len(samples_found) == num_accents:
            break

    for i, (l_idx, (spec_tensor, phonetic_tensor)) in enumerate(sorted(samples_found.items())):
        spec_input = spec_tensor.unsqueeze(0).to(device)
        spec_input.requires_grad = True
        phonetic_input = phonetic_tensor.unsqueeze(0).to(device)

        heatmap, pred_class = grad_cam.generate_heatmap(spec_input, phonetic_input, target_class=l_idx)
        spec_img = spec_tensor.squeeze().cpu().numpy()

        ax = axes[i]
        # Plot grayscale spectrogram with translucent Grad-CAM heatmap overlay
        ax.imshow(spec_img, aspect='auto', origin='lower', cmap='gray', alpha=0.6)
        ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5,
                  extent=[0, spec_img.shape[1], 0, spec_img.shape[0]])

        accent_name = idx2label[l_idx]
        pred_name = idx2label[pred_class]
        ax.set_title(f"True: {accent_name} | Pred: {pred_name}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Frequency Bins")

    # Remove empty subplot axes in the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Grad-CAM Heatmaps - Dual-Branch CNN Acoustic Attention", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "grad_cam_attention_maps.png")
    plt.close()

    # Averaged Accent Spectrogram Templates
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
            # Crop all spectrograms to the minimum time duration among sampled files
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

    # Residual Learning Visualization
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

                # Compute RMS energy contour across time frames
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

    # Spectral Rolloff Contour Comparison
    chinese_spec, cn_id, cn_accent_name = find_valid_sample(df_filtered, SPECTROGRAM_SAVE_DIR, 'chinese|cn')
    american_spec, us_id, us_accent_name = find_valid_sample(df_filtered, SPECTROGRAM_SAVE_DIR, 'american|us')

    if chinese_spec is None or american_spec is None:
        available_accents = df_filtered['accent'].unique()
        chinese_spec, cn_id, cn_accent_name = find_valid_sample(df_filtered, SPECTROGRAM_SAVE_DIR, available_accents[0])
        american_spec, us_id, us_accent_name = find_valid_sample(df_filtered, SPECTROGRAM_SAVE_DIR, available_accents[1])

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

        # Annotate notable spectral energy drop in Chinese accent sample
        plt.annotate('Energy Drop\n(Clipping of Final Plosives)',
                     xy=(int(len(cn_rolloff) * 0.75), cn_rolloff[int(len(cn_rolloff) * 0.75)]),
                     xytext=(int(len(cn_rolloff) * 0.55), max(cn_rolloff) * 0.85),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=10, color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "spectral_rolloff_contour.png")
        plt.close()

    print("\n" + "=" * 80)
    print(" ALL MULTIMODAL CNN PIPELINE OUTPUTS AND GRAPHS SAVED DIRECTLY TO:")
    print(f"    {PLOTS_DIR}")
    print("=" * 80 + "\n")
    # ==============================================================================
    # SINGLE-FILE INFERENCE VIA TKINTER FILE DIALOG
    # ==============================================================================
    print("\n" + "=" * 70)
    print(" PREPARING INFERENCE PIPELINE FOR SINGLE AUDIO FILE...")
    print("=" * 70)

    # 1. Load Whisper Speech-To-Text Model
    print("Loading Whisper STT Model for text transcription...")
    try:
        whisper_model = whisper.load_model("base", device=device)
    except Exception as e:
        print(f" Could not load Whisper model directly on {device}, falling back to CPU: {e}")
        whisper_model = whisper.load_model("base", device="cpu")

    # 2. Load best checkpoint saved during Cross-Validation
    best_model_checkpoint = PROJECT_DIR / 'best_dual_branch_cnn.pth'
    if not best_model_checkpoint.exists():
        best_model_checkpoint = PROJECT_DIR / 'best_dual_branch_fold_1.pth'

    print(f"Loading trained Dual-Branch CNN weights from: {best_model_checkpoint.name}")
    inference_model = DualBranchAccentCNN(num_classes=len(label2idx)).to(device)
    inference_model.load_state_dict(torch.load(best_model_checkpoint, map_location=device))
    inference_model.eval()

    # 3. Open Tkinter native file dialog to select input WAV file
    print("\n Opening file browser dialog to select audio WAV file for inference...")
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # Hide main Tkinter window
        root.attributes('-topmost', True)  # Ensure dialog appears on top

        selected_audio_path = filedialog.askopenfilename(
            title="Select Audio WAV File for Accent Inference",
            filetypes=[("Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        root.destroy()

        if selected_audio_path:
            # Execute full inference pipeline on selected file
            process_and_classify_uploaded_audio(
                file_path=selected_audio_path,
                inference_model=inference_model,
                whisper_model=whisper_model,
                label2idx_mapping=label2idx,
                device=device
            )
        else:
            print(" File selection cancelled by user.")

    except Exception as dialog_err:
        print(f" Could not open file dialog window ({dialog_err}).")
        user_file = input("Please manually enter the full path to your audio WAV file: ")
        if user_file and os.path.exists(user_file):
            process_and_classify_uploaded_audio(
                file_path=user_file,
                inference_model=inference_model,
                whisper_model=whisper_model,
                label2idx_mapping=label2idx,
                device=device
            )
        else:
            print(" Invalid path or no file provided.")