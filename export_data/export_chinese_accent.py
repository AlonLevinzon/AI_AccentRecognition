import csv
import io
import os
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Audio

# Silence the Symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Path definitions
OUTPUT_DIR = Path("./ascend_english_dataset")
OUTPUT_CSV = Path("./ascend_english_metadata.csv")

# Create output directory for audio files if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define filtering constraints
MAX_TOTAL_SAMPLES = 1000
MAX_SAMPLES_PER_SPEAKER = 10


def download_and_generate_metadata():
    # Define the splits we want to collect from
    splits = ["train", "validation", "test"]

    # Track counts and metadata globally across all splits
    speaker_counts = defaultdict(int)
    total_saved = 0
    metadata_rows = []

    print("Loading and processing CAiRE/ASCEND dataset across all splits (train, validation, test)...")

    for split_name in splits:
        if total_saved >= MAX_TOTAL_SAMPLES:
            break

        print(f"\n--- Processing split: '{split_name}' ---")
        try:
            # Load the specific split
            dataset = load_dataset("CAiRE/ASCEND", split=split_name)
            # Disable decoding to bypass torchcodec/FFmpeg crashes
            dataset = dataset.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"Failed to load split {split_name}: {e}")
            continue

        for idx, sample in enumerate(dataset):
            # Stop if we reached the overall limit of 1000 total audio samples
            if total_saved >= MAX_TOTAL_SAMPLES:
                print(f"Reached the global limit of {MAX_TOTAL_SAMPLES} total samples.")
                break

            # CRITICAL FILTER: Keep ONLY samples where language is strictly 'en'
            lang = sample.get("language") or ""
            if lang.strip().lower() != "en":
                continue

            # Extract speaker ID (falls back to original_speaker_id or a generic ID if empty)
            raw_speaker_id = sample.get("speaker_id") or sample.get("original_speaker_id")
            if raw_speaker_id is None:
                speaker_id = f"spk_{split_name}_{idx}"
            else:
                speaker_id = f"spk_{raw_speaker_id}"

            # Get transcription
            text = sample.get("transcription") or ""
            text = text.strip()

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
                print(f"Saved {total_saved}/{MAX_TOTAL_SAMPLES} English-only samples so far...")

    # Write the CSV file with the exact 3 requested columns
    headers = ["filename", "speaker_id", "text"]
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n--- Final Summary ---")
    print(f"Successfully processed and saved {total_saved} English-only ASCEND audio files to: {OUTPUT_DIR.resolve()}")
    print(f"Total unique speakers collected across all splits: {len(speaker_counts)}")
    print(f"Metadata file saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    download_and_generate_metadata()