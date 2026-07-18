import csv
import io
import os
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Silence the Symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Path definitions
OUTPUT_DIR = Path("./us_common_voice")
OUTPUT_CSV = Path("./us_common_voice_metadata.csv")

# Create output directory for audio files if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define filtering constraints
MAX_TOTAL_SAMPLES = 1000
MAX_SAMPLES_PER_SPEAKER = 10


def download_and_generate_metadata():
    print("Loading Common Voice dataset in streaming mode...")

    # SAFE & MODERN SOURCE: 'fixie-ai/common_voice_17_0' is fully converted to Parquet format.
    # No loading scripts (.py files) are used, which avoids any security blocks.
    # We do NOT use 'trust_remote_code' as it is no longer required or supported here.
    dataset = load_dataset(
        "fixie-ai/common_voice_17_0",
        "en",
        split="train",
        streaming=True
    )

    # SAFE BYPASS: Disable automatic audio decoding to avoid torchcodec/FFmpeg dependencies on Windows
    dataset = dataset.decode(False)

    # Track counts and metadata
    speaker_counts = defaultdict(int)
    total_saved = 0
    metadata_rows = []

    print("Searching and processing US Accent audio samples...")
    for idx, sample in enumerate(dataset):
        # Stop if we reached the limit of 1000 total audio samples
        if total_saved >= MAX_TOTAL_SAMPLES:
            print(f"Reached the limit of {MAX_TOTAL_SAMPLES} total samples.")
            break

        # Extract declared accent from Common Voice metadata fields
        accent = sample.get("accent", "") or ""

        if not accent and "accents" in sample:
            accent = sample.get("accents", "")

        # Standardize search (checking for 'united states', 'us', 'usa', etc.)
        if not accent or ("united states" not in accent.lower() and accent.lower() != "us"):
            continue

        # Extract speaker ID (represented by 'client_id' in Common Voice)
        raw_client_id = sample.get("client_id") or f"client_{idx}"
        speaker_id = raw_client_id[:12]  # Keep first 12 chars for cleaner filenames

        # Extract sentence/transcription (Common Voice uses the 'sentence' column)
        text = sample.get("sentence") or ""

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

        # Define a unique filename
        filename = f"{speaker_id}_{speaker_counts[speaker_id]:02d}.wav"
        file_path = OUTPUT_DIR / filename

        # Decode raw bytes in memory and save as WAV
        try:
            with io.BytesIO(raw_bytes) as byte_io:
                audio_data, sampling_rate = sf.read(byte_io)
            sf.write(file_path, audio_data, sampling_rate)
        except Exception as e:
            print(f"Failed to decode or save {filename}: {e}")
            continue

        # Append metadata
        metadata_rows.append({
            "filename": filename,
            "speaker_id": speaker_id,
            "text": text
        })

        # Update counters
        speaker_counts[speaker_id] += 1
        total_saved += 1

        if total_saved % 50 == 0:
            print(f"Saved {total_saved}/{MAX_TOTAL_SAMPLES} US accent samples...")

    # Write the CSV file
    headers = ["filename", "speaker_id", "text"]
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n--- Summary ---")
    print(f"Successfully processed and saved {total_saved} US English audio files to: {OUTPUT_DIR.resolve()}")
    print(f"Total unique speakers collected: {len(speaker_counts)}")
    print(f"Metadata file saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    download_and_generate_metadata()