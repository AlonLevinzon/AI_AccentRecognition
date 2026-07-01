import os
import io
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

# 1. Configuration
DATASET_NAME = "openslr/librispeech_asr"
CONFIG_NAME = "clean"
SPLIT_NAME = "train.100"  # FIXED: Changed from 'train.clean.100' to 'train.100'

OUTPUT_DIR = "librispeech_extracted"
METADATA_FILE = os.path.join(OUTPUT_DIR, "usa_metadata.xlsx")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_files")

os.makedirs(AUDIO_DIR, exist_ok=True)

# 2. Load dataset with STREAMING mode enabled
print(f"Streaming subset {SPLIT_NAME} from {DATASET_NAME} ({CONFIG_NAME})...")
dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME, streaming=True)

# Disable automatic audio decoding to fetch raw bytes
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. Stream and extract raw metadata records
print("Streaming metadata records from Hugging Face...")
raw_records = []

for idx, item in enumerate(dataset):
    raw_records.append({
        "file": item.get("file") or "",
        "text": item.get("text") or "",
        "speaker_id": int(item.get("speaker_id", 0)),
        "chapter_id": int(item.get("chapter_id", 0)),
        "id": item.get("id") or f"ls_{idx}",
        "audio_bytes": item["audio"]["bytes"]
    })

    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1} rows...")

df_raw = pd.DataFrame(raw_records)

# 4. Filter and balance: Max 10 samples per speaker
print("Balancing dataset (max 10 samples per unique speaker)...")
df_balanced = (
    df_raw.groupby("speaker_id", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 10), random_state=42))
)

# 5. Global cap: Limit total dataset size to 2000 audio clips
if len(df_balanced) > 2000:
    print(f"Capping overall dataset size down to 2000 samples...")
    df_balanced = df_balanced.sample(n=2000, random_state=42).reset_index(drop=True)

# 6. Save selected audio files locally
print(f"Saving selected audio files to {AUDIO_DIR}...")
final_records = []

for _, row in df_balanced.iterrows():
    audio_bytes = row["audio_bytes"]
    local_audio_rel_path = f"audio_files/{row['id']}.wav"
    destination_path = os.path.join(OUTPUT_DIR, local_audio_rel_path)

    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        sf.write(destination_path, data, samplerate)
    except Exception as e:
        continue

    final_records.append({
        "file": local_audio_rel_path,
        "audio": str({"path": local_audio_rel_path, "sampling_rate": samplerate}),
        "text": row["text"],
        "speaker_id": row["speaker_id"],
        "chapter_id": row["chapter_id"],
        "id": row["id"]
    })

# 7. Convert to DataFrame and save to Excel sheet
df_final = pd.DataFrame(final_records)
with pd.ExcelWriter(METADATA_FILE, engine="openpyxl") as writer:
    df_final.to_excel(writer, sheet_name="metadata", index=False)

print("\n--- Process Finished Successfully via Streaming ---")
print(f"Total files saved: {len(df_final)}")
print(f"Metadata file successfully generated at: {METADATA_FILE}")