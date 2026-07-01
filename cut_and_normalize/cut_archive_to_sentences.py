import os
import pandas as pd
from pathlib import Path
# Import the existing sentence extractor function from your script
from sentences_extractor import cut_audio_to_sentences


def generate_accent_metadata():
    # 1. Dynamically find the directory where pipeline_manager.py is stored
    current_script_dir = Path(__file__).resolve().parent

    # 2. Go one folder up from cut_and_normalize (to the root directory) to find the CSV
    metadata_csv_path = current_script_dir.parent / "metadata_archive.xlsx"

    # 3. Target the output folder inside the root directory
    base_output_path = current_script_dir.parent / "sentences_samples"

    # Check if the file exists using the calculated absolute path
    if not metadata_csv_path.exists():
        print(f"Error: Master metadata file not found at {metadata_csv_path.resolve()}")
        return

    print("Loading master metadata...")
    df_master = pd.read_excel(metadata_csv_path)

    # Clean up filename column to ensure safe comparison matching
    df_master['filename_stem'] = df_master['filename'].apply(lambda x: Path(str(x)).stem.lower())

    print("Running audio sentence extraction...")
    # Trigger your original splitting logic
    cut_audio_to_sentences()

    print("Generating per-accent metadata files...")
    if not base_output_path.exists():
        print(f"Error: Output directory {base_output_path} does not exist.")
        return

    # Iterate through each accent folder inside the target sentences directory
    for accent_folder in os.listdir(base_output_path):
        accent_dir_path = base_output_path / accent_folder
        if not accent_dir_path.is_dir():
            continue

        print(f"Processing metadata for accent: {accent_folder}")
        accent_rows = []

        # Iterate through sentence sub-folders (e.g., sentence_0, sentence_1)
        for sentence_folder in os.listdir(accent_dir_path):
            sentence_dir_path = accent_dir_path / sentence_folder
            if not sentence_dir_path.is_dir() or not sentence_folder.startswith("sentence_"):
                continue

            sentence_index = sentence_folder.split("_")[1]

            # Find all sliced wav files within this specific sentence directory
            for wav_file in sentence_dir_path.glob("*.wav"):
                # Extract the original filename stem using your file naming convention
                file_stem_orig = wav_file.stem.split("_part_")[0].lower()

                # Search for the speaker details in the master metadata mapping
                matched_row = df_master[df_master['filename_stem'] == file_stem_orig]

                if not matched_row.empty:
                    # Clone the original metadata record
                    new_row = matched_row.iloc[0].copy()

                    # Inject runtime features for future feature extraction pipelines
                    new_row['sentence_index'] = sentence_index
                    new_row['split_filename'] = wav_file.name
                    new_row['relative_audio_path'] = f"./{sentence_folder}/{wav_file.name}"

                    accent_rows.append(new_row)
                else:
                    print(f"Warning: No metadata found for original file stem: {file_stem_orig}")

        # If data rows exist for this accent, save the specialized metadata CSV
        if accent_rows:
            df_accent = pd.DataFrame(accent_rows)

            # Drop the temporary lookup column before saving
            if 'filename_stem' in df_accent.columns:
                df_accent = df_accent.drop(columns=['filename_stem'])

            output_csv_name = accent_dir_path / f"{accent_folder}_metadata.csv"
            df_accent.to_csv(output_csv_name, index=False, encoding='utf-8-sig')
            print(f"Saved: {output_csv_name} (Total rows: {len(df_accent)})")


if __name__ == "__main__":
    print("Starting Pipeline Manager...")
    generate_accent_metadata()
    print("Pipeline completed successfully!")