import os
import pandas as pd
import soundfile as sf
from pathlib import Path

# Updated Import Statement: Importing from the 'cut_and_normalize' subfolder
from cut_and_normalize.audio_normalization import process_raw_audio_pipeline


def process_and_save_audio(src_file_path, dest_dir_path, target_sr=16000):
    """
    Runs the raw audio through the normalization pipeline and saves it if valid.
    Returns True if the file was processed and saved, False if discarded.
    """
    os.makedirs(dest_dir_path, exist_ok=True)
    dest_file_path = dest_dir_path / src_file_path.name

    # Process the audio via the imported DSP pipeline from cut_and_normalize folder
    processed_audio = process_raw_audio_pipeline(str(src_file_path), target_sr=target_sr)

    if processed_audio is not None:
        # Save the processed numpy array back to a physical .wav file
        sf.write(str(dest_file_path), processed_audio, target_sr)
        return True
    return False


def merge_and_normalize_splits(mode="mixed"):
    """
    Handles dataset unification and audio normalization.
    - mode="mixed": Merges both 'free_text_train_and_test' and 'archive_train_and_test'
    - mode="free_text_only": Processes ONLY 'free_text_train_and_test'
    """
    current_dir = Path(__file__).resolve().parent
    splits = ['train', 'validation', 'test']

    # Define source directories based on your repository structure
    free_text_base = current_dir / "free_text_train_and_test"
    archive_base = current_dir / "archive_train_and_test"

    # Configure target output paths based on selected operational mode
    if mode == "mixed":
        output_base = current_dir / "official_audio_mixed"
        sources_to_read = [free_text_base, archive_base]
        print("\n=============================================")
        print("RUNNING MODE: official_audio_mixed (Full Merge)")
        print("=============================================")
    else:
        output_base = current_dir / "official_audio_free_text_only"
        sources_to_read = [free_text_base]
        print("\n=============================================")
        print("RUNNING MODE: official_audio_free_text_only")
        print("=============================================")

    # Hardcoded list containing the 6 standardized Stella sentences for the Archive dataset split
    stella_sentences = [
        "Please call Stella. Ask her to bring these things",
        "with her from the store: Six spoons of fresh snow peas,",
        "five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "We also need a small plastic snake and a big toy frog for the kids.",
        "She can scoop these things into three red bags,",
        "and we will go meet her Wednesday at the train station."
    ]

    # Tracker dictionaries to count valid samples per split for the final report
    final_counts = {'train': 0, 'validation': 0, 'test': 0}
    split_metadata_accumulators = {split: [] for split in splits}

    # Iterate through each dataset split layout
    for split in splits:
        # Handle variance where one folder might be 'validation' and another 'val'
        split_folder_names = [split, 'val'] if split == 'validation' else [split]

        for src_base in sources_to_read:
            if not src_base.exists():
                continue

            # Check if the active folder layout corresponds to the archive dataset
            is_archive_source = "archive" in src_base.name.lower()

            # Locate the exact split subfolder inside the active dataset base
            active_split_dir = None
            for folder_name in split_folder_names:
                candidate_dir = src_base / folder_name
                if candidate_dir.exists():
                    active_split_dir = candidate_dir
                    break

            if not active_split_dir:
                continue

            # Locate the accompanying metadata CSV file for this specific split
            meta_file_path = None
            for file in os.listdir(active_split_dir):
                if file.endswith('.csv'):
                    meta_file_path = active_split_dir / file
                    break

            if not meta_file_path or not meta_file_path.exists():
                continue

            # Load the corresponding metadata sheet safely
            try:
                df = pd.read_csv(meta_file_path)
                df.columns = df.columns.str.strip().str.lower()
            except Exception as e:
                print(f" -> Error reading {meta_file_path.name}: {e}")
                continue

            # Helper function to fetch standard identifier columns
            def get_first_column(dataframe, alternative_keys):
                for key in alternative_keys:
                    if key in dataframe.columns:
                        return dataframe[key]
                return None

            filename_col = get_first_column(df, ['filename', 'split_filename', 'relative_audio_path'])
            speaker_col = get_first_column(df, ['speaker_id', 'speakerid', 'speaker'])
            accent_col = get_first_column(df, ['accent'])
            text_col = get_first_column(df, ['text', 'transcription', 'sentence'])

            # Row by row validation loop
            for idx, row in df.iterrows():
                if filename_col is None or pd.isna(filename_col.iloc[idx]):
                    continue

                raw_filename = os.path.basename(str(filename_col.iloc[idx]))
                src_audio_file = active_split_dir / raw_filename

                # If file not found in immediate folder, search case-insensitively
                if not src_audio_file.exists():
                    for f in os.listdir(active_split_dir):
                        if f.lower() == raw_filename.lower():
                            src_audio_file = active_split_dir / f
                            raw_filename = f
                            break

                if not src_audio_file.exists():
                    continue

                # Run file through the DSP pipeline and save to target folder if successful
                dest_split_dir = output_base / split
                is_saved = process_and_save_audio(src_audio_file, dest_split_dir)

                if is_saved:
                    final_counts[split] += 1

                    # Initialize text variable to hold final sanitized transcription string
                    final_text = ""

                    # If file originates from archive data or explicitly contains the naming partition flag
                    if is_archive_source or "_part_" in raw_filename.lower():
                        try:
                            # Parse out the segment segment digit from filename structure (e.g., mandarin1_part_3.wav -> 3)
                            part_str = raw_filename.lower().split("_part_")[-1].split(".")[0]
                            part_index = int(part_str)

                            # Safely fetch matching sentence index layout from the hardcoded Stella array
                            if 0 <= part_index < len(stella_sentences):
                                final_text = stella_sentences[part_index]
                        except Exception:
                            # Fallback to structural row string if filename mapping execution fails
                            final_text = text_col.iloc[idx] if text_col is not None else ""
                    else:
                        # Fetch original string natively from CSV sheet for typical free text inputs
                        final_text = text_col.iloc[idx] if text_col is not None else ""

                    # Guarantee that any residual NaN values are safely coerced to blank strings
                    if pd.isna(final_text):
                        final_text = ""

                    # Fetch the raw accent label value safely
                    raw_accent = accent_col.iloc[idx] if accent_col is not None else None

                    # FIX: Enforce centralized accent sanitization mapping 'hindi' target labels to 'indian'
                    if isinstance(raw_accent, str):
                        cleaned_accent = raw_accent.strip().lower()
                        if cleaned_accent == 'hindi':
                            raw_accent = 'Indian'

                    split_metadata_accumulators[split].append({
                        'filename': raw_filename,
                        'speaker_id': speaker_col.iloc[idx] if speaker_col is not None else None,
                        'accent': raw_accent,
                        'text': final_text
                    })

    # Generate and export the newly compiled and cleaned metadata sheets
    print("\nWriting synchronized official metadata sheets...")
    for split in splits:
        if split_metadata_accumulators[split]:
            df_final = pd.DataFrame(split_metadata_accumulators[split])

            # Enforce strict column sequencing layout
            column_order = ['filename', 'speaker_id', 'accent', 'text']
            df_final = df_final[column_order]

            output_meta_path = output_base / split / f"{split}_metadata.csv"
            df_final.to_csv(output_meta_path, index=False, encoding='utf-8-sig')
            print(f" -> Created: {output_meta_path.relative_to(current_dir)} ({len(df_final)} samples)")

    # Print summary execution breakdown statistics report
    print(f"\nExecution Report Summary for [{output_base.name}]:")
    print(f" - TRAIN Split samples:      {final_counts['train']}")
    print(f" - VALIDATION Split samples: {final_counts['validation']}")
    print(f" - TEST Split samples:       {final_counts['test']}")
    print(f" Total clean samples retained: {sum(final_counts.values())}\n")

    return final_counts


if __name__ == "__main__":
    # Task 1: Generate the full combined dataset pipeline (official_audio_mixed)
    mixed_stats = merge_and_normalize_splits(mode="mixed")

    # Task 2: Generate the independent free-text only pipeline (official_audio_free_text_only)
    #free_text_stats = merge_and_normalize_splits(mode="free_text_only")

    print("=========================================================================")
    print("ALL PROCESSING TASKS COMPLETED SUCCESSFULLY!")
    print("=========================================================================")