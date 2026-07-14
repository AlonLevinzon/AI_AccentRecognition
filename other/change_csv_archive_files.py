import os
import re
import pandas as pd
from pathlib import Path


def align_archive_metadata_new():
    # Define paths matching your repository structure
    current_dir = Path(__file__).resolve().parent
    archive_dir = current_dir / "archive_train_and_test"

    splits = ['train', 'validation', 'test']

    if not archive_dir.exists():
        print(f"Error: Archive directory {archive_dir} does not exist.")
        return

    # Standard "Stella" paragraph chunks mapped strictly from part 0 to part 5
    stella_sentences = {
        0: "Please call Stella. Ask her to bring these things",
        1: "with her from the store: Six spoons of fresh snow peas,",
        2: "five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        3: "We also need a small plastic snake and a big toy frog for the kids.",
        4: "She can scoop these things into three red bags,",
        5: "and we will go meet her Wednesday at the train station."
    }

    print("Initializing Archive Metadata Text-Injection & Column Trimming...")

    # Process each split file inside the archive folder
    for split in splits:
        # Input path is the original file
        input_meta_path = archive_dir / f"{split}_metadata.csv"
        # Output path creates a NEW file with 'new_' prefix to prevent overwriting
        output_meta_path = archive_dir / f"new_{split}_metadata.csv"

        if not input_meta_path.exists():
            print(f" -> Warning: {input_meta_path.name} not found. Skipping.")
            continue

        print(f"\n[Processing Archive File] Reading: '{input_meta_path.name}'")

        try:
            # Load the original archive sheet
            df = pd.read_csv(input_meta_path)
            # Normalize column names by removing spaces and forcing lowercase to map safely
            df.columns = df.columns.str.strip().str.lower()
        except Exception as e:
            print(f" -> Error reading {input_meta_path.name}: {e}")
            continue

        # Dynamic mapping to fetch the original content even if naming variant differs
        def get_column_data(dataframe, alternative_keys):
            for key in alternative_keys:
                if key in dataframe.columns:
                    return dataframe[key]
            return None

        # Build the structured, lightweight DataFrame
        df_new = pd.DataFrame()
        df_new['filename'] = get_column_data(df, ['filename', 'split_filename', 'relative_audio_path'])
        df_new['speaker_id'] = get_column_data(df, ['speakerid', 'speaker_id'])
        df_new['accent'] = get_column_data(df, ['accent'])

        # Filter out rows missing core file names
        df_new = df_new.dropna(subset=['filename'])

        # Array to hold the injected texts for this split
        injected_texts = []

        # Iterate over filenames to determine which Stella chunk corresponds to it
        for fname in df_new['filename']:
            fname_str = str(fname)

            # Use Regex to look for a sentence/part number identifier (e.g., _0, _part0, sentence_0)
            part_match = re.search(r'(?:part|sentence|_|^)([0-5])(?:\.wav|\.mp3|\.flac)?$', fname_str, re.IGNORECASE)

            # Fallback search if the previous block wasn't triggered
            if not part_match:
                part_match = re.search(r'([0-5])', fname_str)

            if part_match:
                part_index = int(part_match.group(1))
                # Assign corresponding mapped string sequence
                injected_texts.append(stella_sentences.get(part_index, None))
            else:
                # Fallback if no numeric chunk layout was successfully extracted
                injected_texts.append(None)

        # Allocate the compiled sentences array to the new 'text' column
        df_new['text'] = injected_texts

        # Enforce clean target column sorting layout
        column_order = ['filename', 'speaker_id', 'accent', 'text']
        df_new = df_new[column_order]

        # Save to the NEW file path, keeping your original data completely safe
        try:
            df_new.to_csv(output_meta_path, index=False, encoding='utf-8-sig')
            print(f" -> Successfully created new unified registry: {output_meta_path.name} ({len(df_new)} rows)")
        except Exception as e:
            print(f" -> Error saving new file {output_meta_path.name}: {e}")


if __name__ == "__main__":
    print("=============================================")
    print("Starting Safe Archive Metadata Aligner...     ")
    print("=============================================")
    align_archive_metadata_new()
    print("\nProcess finished! Original metadata remains intact. New files created with 'new_' prefix.")