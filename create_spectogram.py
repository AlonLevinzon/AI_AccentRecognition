import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# Force PyCharm to use soundfile backend matching Colab standards
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# =====================================================================
# GLOBAL CONFIGURATION SETTINGS
# Select RUN_MODE: "free_text", "mixed", or "archive_only"
# =====================================================================
RUN_MODE = "free_text"  # Options: "free_text", "mixed", "archive_only"

CURRENT_DIR = Path(__file__).resolve().parent

# Define directory mapping based on project layout
MODE_DIR_MAP = {
    "free_text": CURRENT_DIR / "machine_learning_official_free_text_only",
    "mixed": CURRENT_DIR / "machine_learning_official_mixed",
    "archive_only": CURRENT_DIR / "machine_learning_official_archive_only"
}

OUTPUT_BASE_DIR = MODE_DIR_MAP.get(RUN_MODE, CURRENT_DIR)

# Spectrogram save directory mapping
SPECTROGRAM_DIRNAME = f"spectrograms_logmel_{RUN_MODE}"
SPECTROGRAM_SAVE_DIR = CURRENT_DIR / SPECTROGRAM_DIRNAME

METADATA_CSV_FILENAME = f"massive_metadata_{RUN_MODE}.csv"
METADATA_CSV_PATH = OUTPUT_BASE_DIR / METADATA_CSV_FILENAME


# =====================================================================
# 1. MEL-SPECTROGRAM EXTRACTION PIPELINE (100% COLAB COMPATIBLE)
# =====================================================================
def process_and_extract_mel_spectrogram(
    audio_path: str, target_sr: int = 16000, target_duration_sec: float = 4.0
) -> torch.Tensor:
    """
    Loads audio, converts to mono, resamples to 16kHz, pads/trims to 4.0s,
    extracts Log-Mel Spectrogram (dB), and applies Z-score normalization.
    Matches Colab execution 1:1.
    """
    waveform, sr = torchaudio.load(audio_path)

    # Convert Stereo to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if necessary
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Enforce fixed target length (4.0 seconds) via padding or trimming
    target_length = int(target_sr * target_duration_sec)
    num_samples = waveform.shape[1]

    if num_samples < target_length:
        pad_amount = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > target_length:
        waveform = waveform[:, :target_length]

    # Compute Mel Spectrogram (n_mels=128, power=2.0)
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=2.0
    )
    mel_spec = mel_transform(waveform)

    # Convert amplitude to Decibels (logarithmic scale)
    mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

    # Apply Z-score standardization
    mean = mel_spec.mean()
    std = mel_spec.std() + 1e-6
    mel_spec = (mel_spec - mean) / std

    return mel_spec


# =====================================================================
# 2. MAIN SPECTROGRAM GENERATION EXECUTION
# =====================================================================
def main():
    print("=" * 70)
    print(f"🚀 PYCHARM SPECTROGRAM GENERATOR INITIALIZED | MODE: [{RUN_MODE.upper()}]")
    print("=" * 70)

    # Verify metadata CSV exists
    if not METADATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"❌ Required metadata file not found at: {METADATA_CSV_PATH}\n"
            f"Please verify dataset path or run previous preprocessing script."
        )

    # Create destination directory
    os.makedirs(SPECTROGRAM_SAVE_DIR, exist_ok=True)

    # Load metadata CSV
    print(f"📂 [LOADING METADATA]: Reading directly from {METADATA_CSV_PATH}")
    metadata_df = pd.read_csv(METADATA_CSV_PATH)

    # Standardize clean file ID for accurate mapping
    metadata_df["clean_id"] = metadata_df["filename"].apply(
        lambda x: os.path.splitext(os.path.basename(str(x)))[0].lower().strip()
    )

    print(f"📋 Loaded {len(metadata_df)} records.")
    print(f"📁 Target Spectrogram Save Path: {SPECTROGRAM_SAVE_DIR}\n")

    generated_count = 0
    skipped_count = 0
    error_count = 0

    # Process audio files
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Generating [{RUN_MODE.upper()}] Spectrograms"):
        file_id = row["clean_id"]
        audio_path = str(row["file_path"])
        save_path = SPECTROGRAM_SAVE_DIR / f"{file_id}.pt"

        # Skip existing
        if save_path.exists():
            skipped_count += 1
            continue

        if not os.path.exists(audio_path):
            print(f"\n⚠️ Audio file missing at: {audio_path}")
            error_count += 1
            continue

        try:
            spec_tensor = process_and_extract_mel_spectrogram(audio_path)
            torch.save(spec_tensor, save_path)
            generated_count += 1
        except Exception as e:
            print(f"\n❌ Error processing {file_id}: {e}")
            error_count += 1

    # =====================================================================
    # 3. REPORT GENERATION
    # =====================================================================
    existing_pt_files = {
        os.path.splitext(f)[0].lower().strip()
        for f in os.listdir(SPECTROGRAM_SAVE_DIR)
        if f.endswith(".pt")
    }
    metadata_df["pt_exists"] = metadata_df["clean_id"].isin(existing_pt_files)

    print("\n" + "=" * 65)
    print(f"📊 SPECTROGRAM GENERATION REPORT [{RUN_MODE.upper()}]")
    print("=" * 65)

    accent_summary = metadata_df.groupby("accent").agg(
        CSV_Total=("clean_id", "count"),
        PT_Generated=("pt_exists", "sum")
    )
    accent_summary["Missing"] = accent_summary["CSV_Total"] - accent_summary["PT_Generated"]
    accent_summary["Success_Rate_%"] = (
        (accent_summary["PT_Generated"] / accent_summary["CSV_Total"]) * 100
    ).round(2)

    print(accent_summary.to_string())
    print("-" * 65)
    print(f"New Generated: {generated_count} | Skipped Existing: {skipped_count} | Errors: {error_count}")
    print(f"Total Matching PT Files Saved: {accent_summary['PT_Generated'].sum()} / {len(metadata_df)}")
    print("=" * 65)


if __name__ == "__main__":
    main()