import csv
import io
import os
import re
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Silence the Symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Path definitions
OUTPUT_DIR = Path("./american_accent_fleurs")
OUTPUT_CSV = Path("./american_metadata.csv")

# Create output directory for audio files if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define filtering constraints
MAX_TOTAL_SAMPLES = 1000
MAX_SAMPLES_PER_SPEAKER = 10


def extract_real_speaker_id(sample, fallback_idx):
    """
    Extracts the true speaker ID (e.g., 'f1932' or 'm0251') from the 'path' column
    or the nested 'audio/path' dictionary.
    """
    # Check the top-level 'path' column shown in your image, or fall back to audio dict path
    audio_path = sample.get("path") or sample.get("audio", {}).get("path") or ""

    if audio_path:
        # Matches patterns like _f1932_chunk_ or _m0521_chunk_
        match = re.search(r'_([fm]\d+)_chunk_', audio_path)
        if match:
            return match.group(1)

    return f"spk_{fallback_idx}"


def download_and_generate_metadata():
    print("Loading dataset google/fleurs with 'en_us' config in streaming mode...")
    # Load the official google/fleurs dataset for American English (en_us)
    dataset = load_dataset("google/fleurs", "en_us", split="train", streaming=True)

    # SAFE BYPASS: Completely disable automatic audio decoding to avoid torchcodec/FFmpeg dependency
    dataset = dataset.decode(False)

    # Track counts and metadata
    speaker_counts = defaultdict(int)
    total_saved = 0
    metadata_rows = []

    print("Processing audio samples...")
    for idx, sample in enumerate(dataset):
        # Stop if we reached the limit of 1000 total audio samples
        if total_saved >= MAX_TOTAL_SAMPLES:
            print(f"Reached the limit of {MAX_TOTAL_SAMPLES} total samples.")
            break

        # Extract the TRUE speaker ID from the path column
        speaker_id = extract_real_speaker_id(sample, idx)

        # Get transcription (using the exact 'transcription' column from your image)
        text = sample.get("transcription") or ""

        # Skip sample if text is empty
        if not text:
            continue

        # Check if we reached the limit of 10 samples for this specific speaker
        if speaker_counts[speaker_id] >= MAX_SAMPLES_PER_SPEAKER:
            continue

        # Extract the raw audio data dictionary
        audio_dict = sample.get("audio")
        if not audio_dict:
            continue

        # Get the raw bytes from the non-decoded dictionary
        raw_bytes = audio_dict.get("bytes")
        if raw_bytes is None:
            continue

        # Define a unique filename using the clean speaker ID
        filename = f"{speaker_id}_{speaker_counts[speaker_id]:02d}.wav"
        file_path = OUTPUT_DIR / filename

        # Decode raw bytes using soundfile and save
        try:
            with io.BytesIO(raw_bytes) as byte_io:
                audio_data, sampling_rate = sf.read(byte_io)
            sf.write(file_path, audio_data, sampling_rate)
        except Exception as e:
            print(f"Failed to decode or save {filename}: {e}")
            continue

        # Append metadata in the exact format: filename, speaker_id, text
        metadata_rows.append({
            "filename": filename,
            "speaker_id": speaker_id,
            "text": text
        })

        # Update counters
        speaker_counts[speaker_id] += 1
        total_saved += 1

        if total_saved % 100 == 0:
            print(f"Saved {total_saved}/{MAX_TOTAL_SAMPLES} samples...")

    # Write the CSV file
    headers = ["filename", "speaker_id", "text"]
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n--- Summary ---")
    print(f"Successfully processed and saved {total_saved} American English audio files to: {OUTPUT_DIR.resolve()}")
    print(f"Total unique speakers collected: {len(speaker_counts)}")
    print(f"Metadata file saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    download_and_generate_metadata()