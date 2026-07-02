import os
import shutil
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset():
    # Setup paths matching your structure
    current_dir = Path(__file__).resolve().parent
    base_input_path = current_dir / "sentences_samples"
    base_output_path = current_dir / "archive_train_and_test"

    splits = ['train', 'val', 'test']

    # Create the target directory structure
    for split in splits:
        os.makedirs(base_output_path / split, exist_ok=True)

    # Accumulators to hold rows for the new split metadata sheets
    metadata_accumulators = {split: [] for split in splits}

    if not base_input_path.exists():
        print(f"Error: Base directory {base_input_path} does not exist.")
        return

    print("Starting fallback-safe dataset split process...")

    # Iterate through each accent folder (e.g., usa, nigerian, hindi, chinese)
    for accent_folder in os.listdir(base_input_path):
        accent_dir = base_input_path / accent_folder
        if not accent_dir.is_dir():
            continue

        print(f"\n[Processing Accent] Grouping speakers for: '{accent_folder}'")

        # Load original accent metadata sheet to a list of dicts
        main_meta_path = accent_dir / f"{accent_folder}_metadata.csv"
        meta_rows = []
        if main_meta_path.exists():
            try:
                df_accent_meta = pd.read_csv(main_meta_path)
                df_accent_meta.columns = df_accent_meta.columns.str.strip()  # Clean column names
                meta_rows = df_accent_meta.to_dict(orient='records')
            except Exception as e:
                print(f" -> Warning: Could not read metadata file {main_meta_path.name}: {e}")
        else:
            print(f" -> Warning: Metadata file {main_meta_path.name} not found.")

        # Map to group all audio chunks by their unique speaker ID
        speaker_to_files = {}

        # Scan through all sentence folders (sentence_0 to sentence_5)
        for sentence_folder in os.listdir(accent_dir):
            sentence_dir = accent_dir / sentence_folder
            if not sentence_dir.is_dir() or not sentence_folder.startswith("sentence_"):
                continue

            for file_name in os.listdir(sentence_dir):
                if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
                    # Extract speaker ID using regex (everything before '_part')
                    speaker_match = re.match(r"^([a-zA-Z0-9]+)_part", file_name)
                    if speaker_match:
                        speaker = speaker_match.group(1)
                        file_path = sentence_dir / file_name

                        if speaker not in speaker_to_files:
                            speaker_to_files[speaker] = []
                        speaker_to_files[speaker].append((file_name, file_path))

        # List of unique speakers for the current accent
        all_speakers = list(speaker_to_files.keys())
        if not all_speakers:
            continue

        # Split speakers strictly to prevent Data Leakage (70% / 15% / 15%)
        train_speakers, temp_speakers = train_test_split(all_speakers, test_size=0.30, random_state=42)
        val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.50, random_state=42)

        # --- THE SPEAKER-LEVEL CAP FOR USA ACCENT ---
        # If the current accent is 'usa', strictly slice the speaker lists to the requested speaker counts
        if accent_folder == 'usa':
            train_speakers = train_speakers[:50]
            val_speakers = val_speakers[:10]
            test_speakers = test_speakers[:10]
            print(f" -> [USA Speaker Cap Applied] Retained exact speakers: Train=50, Val=10, Test=10")

        # Create a mapping dictionary for quick lookup during file distribution
        speaker_split_map = {}
        for sp in train_speakers: speaker_split_map[sp] = 'train'
        for sp in val_speakers: speaker_split_map[sp] = 'val'
        for sp in test_speakers: speaker_split_map[sp] = 'test'

        # Track how many total files are actually processed for reporting
        files_processed_count = {'train': 0, 'val': 0, 'test': 0}

        # Copy files to flat directory structures and assemble metadata rows
        for speaker, files in speaker_to_files.items():
            # Skip if the speaker did not make it into the sliced maps (relevant for USA capping)
            if speaker not in speaker_split_map:
                continue

            target_split = speaker_split_map[speaker]
            target_dir = base_output_path / target_split

            for file_name, src_path in files:
                dest_path = target_dir / file_name
                files_processed_count[target_split] += 1

                # Physical copy directly into the flat split folder
                shutil.copy2(src_path, dest_path)

                # 1. Build a base fallback dictionary containing ALL columns found in the original sheet
                row_data = {'filename': file_name, 'accent': accent_folder, 'speaker': speaker}
                if meta_rows:
                    for key in meta_rows[0].keys():
                        if key not in row_data:
                            row_data[key] = None

                # 2. Search for the exact or closest file row match in the original metadata
                file_stem = src_path.stem
                matched_attributes = {}

                for row in meta_rows:
                    orig_filename = str(row.get('filename', '')).strip()
                    if (orig_filename == file_name or
                            orig_filename == file_stem or
                            orig_filename in file_name or
                            file_stem in orig_filename):
                        matched_attributes = row
                        break

                # 3. CRITICAL FALLBACK: If the specific file chunk isn't listed, find ANY row belonging to this speaker
                if not matched_attributes and meta_rows:
                    for row in meta_rows:
                        orig_filename = str(row.get('filename', '')).strip()
                        if speaker in orig_filename or row.get('speaker') == speaker:
                            matched_attributes = row
                            break

                # 4. Merge the found data back into our row dictionary safely
                if matched_attributes:
                    for k, v in matched_attributes.items():
                        if k != 'filename':  # Maintain our clean destination filename
                            row_data[k] = v

                metadata_accumulators[target_split].append(row_data)

        print(
            f" -> Allocation breakdown: Train={len(train_speakers)} speakers ({files_processed_count['train']} files), "
            f"Val={len(val_speakers)} speakers ({files_processed_count['val']} files), "
            f"Test={len(test_speakers)} speakers ({files_processed_count['test']} files)")

    # Generate and save independent metadata files for each split directory
    print("\nWriting consolidated metadata sheets...")
    for split in splits:
        if metadata_accumulators[split]:
            df_split = pd.DataFrame(metadata_accumulators[split])

            # Clean up potential column remnants from merging logic
            df_split = df_split.drop(columns=['split', 'join_key', 'filename_stem'], errors='ignore')

            # Enforce priority structural identifiers to be the leftmost columns
            priority_cols = ['filename', 'accent', 'speaker', 'age', 'sex']
            existing_priority = [c for c in priority_cols if c in df_split.columns]
            other_cols = [c for c in df_split.columns if c not in existing_priority]
            df_split = df_split[existing_priority + other_cols]

            output_meta_path = base_output_path / split / f"{split}_metadata.csv"
            df_split.to_csv(output_meta_path, index=False, encoding='utf-8-sig')
            print(f" -> Successfully created: {output_meta_path.relative_to(current_dir)} ({len(df_split)} records)")


if __name__ == "__main__":
    print("=============================================")
    print("Initializing Robust Fallback Splitter...")
    print("=============================================")
    split_dataset()
    print("\nProcess finished! Metadata maps are fully populated.")