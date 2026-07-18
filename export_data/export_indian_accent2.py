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
OUTPUT_DIR = Path("./indian_fleurs_accent")
OUTPUT_CSV = Path("./metadata.csv")

# Create output directory for audio files if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define filtering constraints
MAX_TOTAL_SAMPLES = 1500
MAX_SAMPLES_PER_SPEAKER = 10


def extract_real_speaker_id(sample, fallback_idx):
    """
    Extracts the true speaker ID (e.g., 'f1932' or 'm0251') directly from the
    original audio path/filename inside the audio dictionary.
    """
    audio_dict = sample.get("audio", {})
    audio_path = audio_dict.get("path") or ""

    if audio_path:
        # Regex to find patterns like _f1932_chunk_ or _m0521_chunk_
        match = re.search(r'_([fm]\d+)_chunk_', audio_path)
        if match:
            return match.group(1)

    return f"spk_{fallback_idx}"


def clean_transcription_text(text):
    """
    Cleans the transcription text by removing residual audio paths and tab characters
    (e.g., stripping 'audio/281474976879627_f1171_chunk_0.wav \t 72262 \t' from the actual text).
    """
    if not text:
        return ""

    # Standardize string
    text_str = str(text).strip()

    # Regex pattern to catch leading audio path followed by numbers and tabs
    # Matches anything ending with .wav, followed by optional numbers and tabs
    cleaned_text = re.sub(r'^.*?\.wav\s*\d*\s*\t*', '', text_str, flags=re.IGNORECASE)

    # Final strip to clean up residual whitespaces or tabs
    return cleaned_text.strip()


def download_and_generate_metadata():
    print("Loading dataset Gbssreejith/indian-english-voice in streaming mode...")
    # Load dataset in streaming mode
    dataset = load_dataset("Gbssreejith/indian-english-voice", split="train", streaming=True)

    # SAFE BYPASS: Completely disable automatic audio decoding to avoid torchcodec/FFmpeg dependency
    dataset = dataset.decode(False)

    # Track counts and metadata
    speaker_counts = defaultdict(int)
    total_saved = 0
    metadata_rows = []

    print("Processing audio samples...")
    for idx, sample in enumerate(dataset):
        # Stop if we reached the limit of 1500 total audio samples
        if total_saved >= MAX_TOTAL_SAMPLES:
            print(f"Reached the limit of {MAX_TOTAL_SAMPLES} total samples.")
            break

        # Extract the TRUE speaker ID from the filename path
        speaker_id = extract_real_speaker_id(sample, idx)

        # Get transcription and CLEAN it from any embedded path artifacts
        raw_text = sample.get("transcription") or sample.get("text") or ""
        text = clean_transcription_text(raw_text)

        # Skip sample if text is invalid after cleaning
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

        # Define a unique filename using the clean speaker ID extracted from the path
        filename = f"{speaker_id}_{speaker_counts[speaker_id]:02d}.wav"
        file_path = OUTPUT_DIR / filename

        # Decode the raw WAV/FLAC bytes in memory using soundfile and save
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

    # Write the CSV file with the exact 3 requested columns
    headers = ["filename", "speaker_id", "text"]
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n--- Summary ---")
    print(f"Successfully processed and saved {total_saved} audio files to: {OUTPUT_DIR.resolve()}")
    print(f"Total unique speakers collected: {len(speaker_counts)}")
    print(f"Metadata file saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    download_and_generate_metadata()