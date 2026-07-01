import os
import io
import csv
import soundfile as sf
from datasets import load_dataset, Audio
from collections import defaultdict

# Set Hugging Face cache directory to drive D
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache"

print("Loading facebook/voxpopuli for English subset...")
# Load the English subset where non-native accents are located
dataset = load_dataset("facebook/voxpopuli", "en", split="train")

# Enable lazy decoding to optimize RAM usage
dataset = dataset.cast_column("audio", Audio(decode=False))

# Define output directory
output_dir = "more_accents"
os.makedirs(output_dir, exist_ok=True)

metadata_file_path = os.path.join(output_dir, "metadata_voxpopuli_german.csv")
fieldnames = ["file_name", "speaker_id", "text", "gender", "accent"]

# Tracker dictionary to count clips per speaker
speaker_counts = defaultdict(int)
max_clips_per_speaker = 10

total_extracted = 0

print(f"Extracting up to {max_clips_per_speaker} clips per German speaker...")
with open(metadata_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item in dataset:
        # Filter by native language (de = German)
        if item.get("native_language") == "de":
            speaker_id = item.get("speaker_id")

            # Skip speaker if the limit of 10 clips is reached
            if speaker_counts[speaker_id] >= max_clips_per_speaker:
                continue

            # Generate a structured unique filename
            clip_index = speaker_counts[speaker_id]
            file_name = f"voxpopuli_german_{speaker_id}_{clip_index}.wav"
            file_path = os.path.join(output_dir, file_name)

            try:
                # Extract bytes and convert to physical WAV file
                audio_bytes = item["audio"]["bytes"]
                audio_buffer = io.BytesIO(audio_bytes)
                audio_array, sampling_rate = sf.read(audio_buffer)

                # Save the raw audio file
                sf.write(file_path, audio_array, sampling_rate)

                # Write metadata row
                writer.writerow({
                    "file_name": file_name,
                    "speaker_id": speaker_id,
                    "text": item.get("text", "").strip(),
                    "gender": item.get("gender", "unknown"),
                    "accent": "German"
                })

                # Increment counters
                speaker_counts[speaker_id] += 1
                total_extracted += 1

            except Exception:
                # Skip corrupted audio samples if encountered
                continue

print(f"\nExtraction completed successfully!")
print(f"- Total unique speakers processed: {len(speaker_counts)}")
print(f"- Total German files added to {output_dir}/: {total_extracted}")
print(f"- Metadata saved at: {metadata_file_path}")