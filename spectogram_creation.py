import os
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


# 1. פונקציית החילוץ המעודכנת
def process_and_extract_mel_spectrogram(
    audio_path: str, target_sr: int = 16000, target_duration_sec: float = 4.0
) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Fixed length via padding/trimming (4.0 seconds)
    target_length = int(target_sr * target_duration_sec)
    num_samples = waveform.shape[1]

    if num_samples < target_length:
        pad_amount = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > target_length:
        waveform = waveform[:, :target_length]

    # # Mel Spectrogram (power=1.0 for Linear Amplitude)
    # mel_transform = T.MelSpectrogram(
    #     sample_rate=target_sr,
    #     n_fft=1024,
    #     hop_length=512,
    #     n_mels=128,
    #     power=1.0,
    # )
    # mel_spec = mel_transform(waveform)

    # # Z-score normalization
    # mean = mel_spec.mean()
    # std = mel_spec.std() + 1e-6
    # mel_spec = (mel_spec - mean) / std
    # Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=2.0,  # Power spectrogram
    )
    mel_spec = mel_transform(waveform)

    # המרה לסולם לוגריתמי (Decibels) - משפר דרסטית למידת תדרים ל-CNN רזה
    mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

    # Z-score normalization
    mean = mel_spec.mean()
    std = mel_spec.std() + 1e-6
    mel_spec = (mel_spec - mean) / std
    return mel_spec


# 2. נתיבים והרצת היצירה
RAW_AUDIO_DIR = "/content/drive/MyDrive/project/free_text"
SPECTROGRAM_SAVE_DIR = "/content/drive/MyDrive/project/spectrograms_logmel"
CSV_PATH = (
    "/content/drive/MyDrive/project/master_extracted_features_free_text.csv"
)

os.makedirs(SPECTROGRAM_SAVE_DIR, exist_ok=True)

metadata_df = pd.read_csv(CSV_PATH)
metadata_df["clean_id"] = metadata_df["filename"].apply(
    lambda x: os.path.splitext(os.path.basename(str(x)))[0].lower().strip()
)

valid_extensions = {".wav", ".mp3", ".flac", ".ogg"}
audio_files_map = {}

for root, _, files in os.walk(RAW_AUDIO_DIR):
    for f in files:
        if Path(f).suffix.lower() in valid_extensions:
            file_id = os.path.splitext(f)[0].lower().strip()
            audio_files_map[file_id] = os.path.join(root, f)

print(
    f"נמצאו {len(audio_files_map)} קבצי אודיו בתיקיית המקור '{RAW_AUDIO_DIR}'."
)

for file_id, audio_path in tqdm(
    audio_files_map.items(), desc="Generating Spectrograms"
):
    save_path = os.path.join(SPECTROGRAM_SAVE_DIR, f"{file_id}.pt")
    if os.path.exists(save_path):
        continue
    try:
        spec_tensor = process_and_extract_mel_spectrogram(audio_path)
        torch.save(spec_tensor, save_path)
    except Exception as e:
        print(f"Error processing {file_id}: {e}")

# 3. דוח ספירת קבצים לכל מבטא
existing_pt_files = {
    os.path.splitext(f)[0].lower().strip()
    for f in os.listdir(SPECTROGRAM_SAVE_DIR)
    if f.endswith(".pt")
}
metadata_df["pt_exists"] = metadata_df["clean_id"].isin(existing_pt_files)

print("\n" + "=" * 65)
print('📊 דו"ח ספירת קבצים וספקטרוגרמות לכל מבטא (Accent Summary Report)')
print("=" * 65)

accent_summary = metadata_df.groupby("accent").agg(
    CSV_Total=("clean_id", "count"),
    PT_Generated=("pt_exists", "sum"),
)
accent_summary["Missing"] = (
    accent_summary["CSV_Total"] - accent_summary["PT_Generated"]
)
accent_summary["Success_Rate_%"] = (
    (accent_summary["PT_Generated"] / accent_summary["CSV_Total"]) * 100
).round(2)

print(accent_summary)
print("-" * 65)
print(
    f'סה"כ קבצים ב-CSV: {len(metadata_df)} | סה"כ קבצי PT תואמים שנשמרו: {accent_summary["PT_Generated"].sum()}'
)
print("=" * 65)