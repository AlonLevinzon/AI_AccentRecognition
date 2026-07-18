import os
import re
import io
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

# 1. Configuration
DATASET_NAME = "benjaminogbonna/nigerian_accented_english_dataset" #it's mozilla
OUTPUT_DIR = "nigerian_accent_extracted"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata_huggingface.xlsx")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_files")

os.makedirs(AUDIO_DIR, exist_ok=True)

# 2. Load the dataset from Hugging Face
print(f"Downloading and loading dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split="train")

# Disable automatic decoding to avoid torchcodec/FFmpeg runtime errors
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. Extract raw records keeping the exact original column structure
print("Extracting metadata with original column headers...")
id_pattern = re.compile(r'(ng[fm]_\d+_\d+)')
raw_records = []

for idx, item in enumerate(dataset):
    audio_meta = item.get("audio", {})
    audio_path = audio_meta.get("path") or f"sample_{idx}.wav"
    filename = os.path.basename(audio_path)

    # Isolate the speaker ID for the sampling logic
    match = id_pattern.search(filename)
    if match:
        full_id = match.group(1)
    else:
        full_id = item.get("client_id") or f"unknown_{idx}"

    id_parts = full_id.split('_')
    speaker_id = id_parts[1] if len(id_parts) > 1 else "Unknown"

    # Reconstruct the exact schema from your screenshot
    raw_records.append({
        "audio": f"audio_files/{full_id}.wav",  # Points to the local file we will save
        "client_id": item.get("client_id") or full_id,
        "path": item.get("path") or audio_path,
        "sentence": item.get("sentence") or item.get("text") or "",
        "accent": item.get("accent") or "",
        "locale": item.get("locale") or "",
        "segment": item.get("segment") or "",
        "Speaker ID": speaker_id,  # Temporary column used for balancing
        "Dataset_Index": idx  # Tracking index to pull bytes later
    })

df_raw = pd.DataFrame(raw_records)

# 4. Filter and balance: Keep a maximum of 15 random samples per speaker
print("Balancing dataset (max 15 samples per speaker)...")
df_balanced = (
    df_raw.groupby("Speaker ID", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 15), random_state=42))
)

# 5. Extract and save the corresponding audio files
print(f"Saving selected audio files to {AUDIO_DIR}...")
final_records = []

for _, row in df_balanced.iterrows():
    idx = row["Dataset_Index"]
    item = dataset[int(idx)]

    audio_bytes = item["audio"]["bytes"]
    destination_path = os.path.join(OUTPUT_DIR, row["audio"])

    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        sf.write(destination_path, data, samplerate)
    except Exception as e:
        print(f"Warning: Could not process audio index {idx} due to: {e}")
        continue

    # Save the record keeping only the desired columns
    final_records.append({
        "audio": row["audio"],
        "client_id": row["client_id"],
        "path": row["path"],
        "sentence": row["sentence"],
        "accent": row["accent"],
        "locale": row["locale"],
        "segment": row["segment"]
    })

# 6. Save final metadata sheet with the exact matching columns
df_final = pd.DataFrame(final_records)
with pd.ExcelWriter(METADATA_FILE, engine="openpyxl") as writer:
    df_final.to_excel(writer, sheet_name="metadata", index=False)

print("\n--- Extraction Summary ---")
print(f"Total samples extracted: {len(df_final)}")
print(f"Metadata successfully saved to: {METADATA_FILE} with exact matching headers.")