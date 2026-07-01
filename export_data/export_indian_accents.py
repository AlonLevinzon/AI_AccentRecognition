import os
import io
import csv
import soundfile as sf
from datasets import load_dataset, Audio
from collections import defaultdict

# Set Hugging Face cache directory to drive D
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache"

print("Loading ishands/commonvoice-indian_accent...")
# Load the dataset split
dataset = load_dataset("ishands/commonvoice-indian_accent", split="train")

# Enable lazy decoding to optimize RAM usage
dataset = dataset.cast_column("audio", Audio(decode=False))

# Define output directory
output_dir = "accents_audio/mozzila/indian_accents"
os.makedirs(output_dir, exist_ok=True)

metadata_file_path = os.path.join(output_dir, "indian_metadata.csv")
fieldnames = ["file_name", "speaker_id", "text", "gender", "age", "accent"]

# Tracker dictionary to limit clips per speaker (client_id) up to 15
speaker_counts = defaultdict(int)
max_clips_per_speaker = 15

# Global limit to stop extraction at 2000 total clips
total_extracted = 0
max_total_clips = 2000

print(f"Extracting up to {max_clips_per_speaker} clips per speaker, max total {max_total_clips}...")
with open(metadata_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item in dataset:
        # Stop completely if we reached the global limit of 2000 clips
        if total_extracted >= max_total_clips:
            break

        # Extract unique speaker ID based on the client_id column
        speaker_id = item.get("client_id")
        if not speaker_id:
            continue

        # Skip speaker if the limit of 15 clips is reached
        if speaker_counts[speaker_id] >= max_clips_per_speaker:
            continue

        # Generate a structured unique filename using speaker hash slice
        clip_index = speaker_counts[speaker_id]
        file_name = f"cv_indian_{speaker_id[:15]}_{clip_index}.wav"
        file_path = os.path.join(output_dir, file_name)

        try:
            # Extract audio bytes and convert to physical WAV file
            audio_bytes = item["audio"]["bytes"]
            audio_buffer = io.BytesIO(audio_bytes)
            audio_array, sampling_rate = sf.read(audio_buffer)

            # Save the raw audio file
            sf.write(file_path, audio_array, sampling_rate)

            # Write metadata row using the correct 'sentence' column from your schema
            writer.writerow({
                "file_name": file_name,
                "speaker_id": speaker_id,
                "text": item.get("sentence", "").strip(),
                "gender": item.get("gender", "unknown"),
                "age": item.get("age", "unknown"),
                "accent": "Indian"
            })

            # Increment counters
            speaker_counts[speaker_id] += 1
            total_extracted += 1

            if total_extracted % 100 == 0:
                print(f"Extracted {total_extracted}/{max_total_clips} clips so far...")

        except Exception:
            # Skip corrupted audio samples if encountered
            continue

print(f"\nExtraction completed successfully!")
print(f"- Total unique Indian speakers processed: {len(speaker_counts)}")
print(f"- Total Indian files added to {output_dir}/: {total_extracted}")
print(f"- Metadata saved at: {metadata_file_path}")