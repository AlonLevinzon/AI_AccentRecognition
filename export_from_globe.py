import os
import csv
import io
import soundfile as sf
from datasets import load_dataset, Audio

print("Downloading GLOBE_V3 Dataset...")
dataset = load_dataset("MushanW/GLOBE_V3", split="train")

dataset = dataset.cast_column("audio", Audio(decode=False))

base_output_dir = "globe_v3_by_accents"
os.makedirs(base_output_dir, exist_ok=True)

target_accents = ["spanish", "german", "french", "indian"]
accent_counters = {accent: 0 for accent in target_accents}

csv_files = {}
csv_writers = {}
fieldnames = ["file_name", "id", "text", "accent", "speaker_id", "gender", "duration"]

for accent in target_accents:
    accent_dir = os.path.join(base_output_dir, accent)
    os.makedirs(accent_dir, exist_ok=True)

    csv_path = os.path.join(accent_dir, f"metadata_{accent}.csv")
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    csv_files[accent] = f
    csv_writers[accent] = writer

print("Extracting and saving audio clips by accent folders...")

for idx, item in enumerate(dataset):
    accent_raw = item.get("accent", "")
    accent_lower = accent_raw.lower()

    matched_accent = None
    for target in target_accents:
        if target in accent_lower:
            matched_accent = target
            break

    if matched_accent:
        file_id = item.get("id", idx)
        file_name = f"globe_{file_id}.wav"

        accent_dir = os.path.join(base_output_dir, matched_accent)
        file_path = os.path.join(accent_dir, file_name)

        try:
            audio_bytes = item["audio"]["bytes"]
            audio_buffer = io.BytesIO(audio_bytes)
            audio_array, sampling_rate = sf.read(audio_buffer)

            sf.write(file_path, audio_array, sampling_rate)

            row_data = {
                "file_name": file_name,
                "id": file_id,
                "text": item.get("text", "").strip(),
                "accent": accent_raw,
                "speaker_id": item.get("speaker_id", ""),
                "gender": item.get("gender", ""),
                "duration": item.get("duration", "")
            }
            csv_writers[matched_accent].writerow(row_data)

            accent_counters[matched_accent] += 1

        except Exception as e:
            print(f"Error processing item {file_id}: {e}")
            continue

for f in csv_files.values():
    f.close()

print(f"\n{'=' * 40}")
print("Extraction completed successfully! File counts:")
for accent, count in accent_counters.items():
    print(f" - {accent.capitalize()} accent: {count} files saved")
print(f"{'=' * 40}")
print(f"All datasets organized inside: {base_output_dir}")