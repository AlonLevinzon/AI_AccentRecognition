import os
import shutil
import pandas as pd
from pathlib import Path


def merge_dataset_clean():
    current_dir = Path(__file__).resolve().parent
    source_base = current_dir / "split_train_and_test_by_accent"
    output_base = current_dir / "free_text_train_and_test"

    splits = ['train', 'validation', 'test']

    if not source_base.exists():
        print(f"Error: Source directory {source_base} does not exist.")
        return

    # =========================================================================
    # FUTURE CONFIGURATION PANEL: Change these values to adjust dataset sizes
    # =========================================================================
    max_train_samples = 900  # Hard limit for train samples per accent
    train_ratio = 0.70  # Target ratio for train split (70%)
    val_test_ratio = 0.15  # Target ratio for val and test splits (15% each)
    # =========================================================================

    # Mathematically calculate the limits for validation and test splits dynamically
    total_calculated_samples = round(max_train_samples / train_ratio)
    max_val_samples = round(total_calculated_samples * val_test_ratio)
    max_test_samples = round(total_calculated_samples * val_test_ratio)

    # Dictionary mapping each split to its specific calculated limit
    split_limits = {
        'train': max_train_samples,
        'validation': max_val_samples,
        'test': max_test_samples
    }

    print("=============================================")
    print(f"Calculated limits per accent: Train={max_train_samples}, Val={max_val_samples}, Test={max_test_samples}")
    print("=============================================")

    # Accumulator to hold dataframes for the final split merge
    split_frames = {split: [] for split in splits}

    # Iterate through each accent folder
    for accent_folder in os.listdir(source_base):
        accent_dir = source_base / accent_folder
        if not accent_dir.is_dir():
            continue

        print(f"\n[Processing Accent] '{accent_folder}'")

        for split in splits:
            split_dir = accent_dir / split
            if not split_dir.exists():
                continue

            # Locate metadata file
            meta_file_path = None
            for file in os.listdir(split_dir):
                if file.endswith(('.csv', '.xlsx', '.xls')):
                    meta_file_path = split_dir / file
                    break

            if not meta_file_path or not meta_file_path.exists():
                continue

            # Read dataset safely
            try:
                if meta_file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(meta_file_path)
                else:
                    df = pd.read_csv(meta_file_path)

                df.columns = df.columns.str.strip().str.lower()
            except Exception as e:
                print(f" -> Error reading {meta_file_path.name}: {e}")
                continue

            df_cleaned = pd.DataFrame()

            # Helper function to get first existing column variant
            def get_first_existing_column(dataframe, alternative_keys):
                for key in alternative_keys:
                    if key in dataframe.columns:
                        return dataframe[key]
                return None

            # Standardized column extraction mappings
            if accent_folder == 'chinese':
                df_cleaned['filename'] = get_first_existing_column(df, ['wav_path', 'filename', 'file', 'audio'])
                df_cleaned['speaker_id'] = get_first_existing_column(df,
                                                                     ['speaker_id', 'speaker-id', 'id', 'client_id'])
                df_cleaned['accent'] = 'chinese'
                df_cleaned['text'] = get_first_existing_column(df, ['transcription', 'text', 'sentence'])

            elif accent_folder in ['indian', 'nigerian']:
                df_cleaned['filename'] = get_first_existing_column(df, ['audio', 'filename', 'file', 'wav_path'])
                df_cleaned['speaker_id'] = get_first_existing_column(df, ['client_id', 'speaker_id', 'id'])
                df_cleaned['accent'] = get_first_existing_column(df,
                                                                 ['accent']) if 'accent' in df.columns else accent_folder
                df_cleaned['text'] = get_first_existing_column(df, ['sentence', 'text', 'transcription'])

            elif accent_folder == 'usa':
                df_cleaned['filename'] = get_first_existing_column(df, ['file', 'filename', 'audio', 'wav_path'])
                df_cleaned['speaker_id'] = get_first_existing_column(df,
                                                                     ['speaker-id', 'id', 'speaker_id', 'client_id'])
                df_cleaned['accent'] = 'usa'
                df_cleaned['text'] = get_first_existing_column(df, ['text', 'sentence', 'transcription'])

            if 'filename' in df_cleaned.columns and df_cleaned['filename'] is not None:
                df_cleaned = df_cleaned.dropna(subset=['filename'])
            else:
                continue

            # --- THE SAMPLING LIMITATION BLOCK ---
            # Get the exact hard limit allowed for the current active split type
            current_split_limit = split_limits[split]

            # Slice the dataframe to contain only up to the maximum allowed records
            if len(df_cleaned) > current_split_limit:
                df_cleaned = df_cleaned.head(current_split_limit)
                print(f" -> Notice: Capped data down to {current_split_limit} rows.")

            # Create destination folder and transfer physical files
            dest_split_dir = output_base / split
            os.makedirs(dest_split_dir, exist_ok=True)

            for original_filename in df_cleaned['filename']:
                pure_name = os.path.basename(str(original_filename))
                src_file = split_dir / pure_name

                if src_file.exists():
                    shutil.copy2(src_file, dest_split_dir / pure_name)
                else:
                    for f in os.listdir(split_dir):
                        if f.lower() == pure_name.lower():
                            shutil.copy2(split_dir / f, dest_split_dir / f)
                            break

            # Metadata sanitization and structural cleaning
            df_cleaned['filename'] = df_cleaned['filename'].apply(
                lambda x: os.path.basename(str(x)) if pd.notna(x) else x)
            df_cleaned['speaker_id'] = df_cleaned['speaker_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) and (
                        isinstance(x, float) or (isinstance(x, str) and '.' in str(x))) else (
                str(x).strip() if pd.notna(x) else x))

            split_frames[split].append(df_cleaned)
            print(f" -> Merged {len(df_cleaned)} rows for {accent_folder} ({split})")

    # Compile final standalone output csv sheets
    print("\nWriting final unified metadata sheets...")
    for split in splits:
        if split_frames[split]:
            df_final = pd.concat(split_frames[split], ignore_index=True)

            column_order = ['filename', 'speaker_id', 'accent', 'text']
            df_final = df_final[column_order]

            output_meta_path = output_base / split / f"{split}_metadata.csv"
            df_final.to_csv(output_meta_path, index=False, encoding='utf-8-sig')
            print(f" -> Successfully created: {output_meta_path.relative_to(current_dir)} ({len(df_final)} total rows)")


if __name__ == "__main__":
    merge_dataset_clean()
    print("\nProcess finished! Balances applied accurately.")