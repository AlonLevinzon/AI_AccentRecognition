import os
import math
import shutil
import pandas as pd
from pathlib import Path


def split_dataset():
    # Detect the project root directory safely
    current_dir = Path.cwd()
    if current_dir.name == "cut_and_normalize":
        root_dir = current_dir.parent
    else:
        root_dir = current_dir

    raw_base_path = root_dir / "audio_by_accent_raw"
    output_base_path = root_dir / "split_train_and_test_by_accent"

    TRAIN_PCT = 0.70
    VAL_PCT = 0.15
    TEST_PCT = 0.15

    # Target configuration dictionary per accent with corrected metadata names
    accent_configs = {
        "nigerian": {
            "meta_file": "nigerian_metadata.xlsx",
            "is_excel": True,
            "speaker_col": "client_id",
            "file_col": "audio",
            "keep_cols": ["audio", "client_id", "sentence", "accent"],
            "rename_map": {}
        },
        "indian": {
            "meta_file": "indian_metadata.csv",
            "is_excel": False,
            "speaker_col": "speaker_id",
            "file_col": "file_name",
            "keep_cols": ["audio", "client_id", "sentence", "accent"],
            "rename_map": {"file_name": "audio", "speaker_id": "client_id", "text": "sentence", "accent": "accent"}
        },
        "usa": {
            "meta_file": "usa_metadata.xlsx",
            "is_excel": True,
            "speaker_col": "speaker_id",
            "file_col": "file",
            "keep_cols": ["file", "text", "speaker_id", "id"],
            "rename_map": {}
        }
    }

    if not raw_base_path.exists():
        print(f"Error: Base input directory not found at {raw_base_path.resolve()}")
        return

    print(f"Initializing Substring-Matching Splitter Pipeline from root: {root_dir.resolve()}\n")

    for accent_name, config in accent_configs.items():
        print(f"==================== Processing Accent: {accent_name} ====================")
        accent_raw_dir = raw_base_path / accent_name
        meta_path = raw_base_path / config["meta_file"]

        if not meta_path.exists():
            print(f"Warning: Metadata file missing for {accent_name} at {meta_path.resolve()}. Skipping.")
            continue

        # Load metadata sheet
        if config["is_excel"]:
            df = pd.read_excel(meta_path)
        else:
            df = pd.read_csv(meta_path)

        speaker_col = config["speaker_col"]
        if speaker_col not in df.columns:
            print(f"Error: Target column '{speaker_col}' not found in registry. Skipping.")
            continue

        # Get list of all actual physical files in the directory
        disk_files = []
        if accent_raw_dir.exists():
            disk_files = [f for f in os.listdir(accent_raw_dir) if os.path.isfile(accent_raw_dir / f)]
        else:
            print(f"Warning: Physical audio folder missing at {accent_raw_dir.resolve()}")

        # Step 1: Extract unique speakers maintaining chronological row order from top to bottom
        unique_speakers = []
        seen = set()
        for spk in df[speaker_col]:
            if pd.notna(spk) and spk not in seen:
                seen.add(spk)
                unique_speakers.append(spk)

        total_speakers = len(unique_speakers)
        print(f"Total Unique Speakers found: {total_speakers}")

        if total_speakers == 0:
            print("No valid records discovered. Skipping.")
            continue

        # Step 2: Calculate deterministic splitting borders
        train_end_idx = math.ceil(total_speakers * TRAIN_PCT)
        val_end_idx = train_end_idx + math.ceil(total_speakers * VAL_PCT)

        train_speakers = set(unique_speakers[:train_end_idx])
        val_speakers = set(unique_speakers[train_end_idx:val_end_idx])
        test_speakers = set(unique_speakers[val_end_idx:])

        print(
            f"Split Count -> Train: {len(train_speakers)}, Validation: {len(val_speakers)}, Test: {len(test_speakers)}")

        groups = {
            "train": train_speakers,
            "validation": val_speakers,
            "test": test_speakers
        }

        # Step 3: Filter rows, copy files, apply transformations and save out clean dataframes
        for group_name, group_speaker_set in groups.items():
            if not group_speaker_set:
                continue

            df_group = df[df[speaker_col].isin(group_speaker_set)].copy()
            group_output_dir = output_base_path / accent_name / group_name
            group_output_dir.mkdir(parents=True, exist_ok=True)

            copied_files_count = 0
            original_file_col = config["file_col"]

            if disk_files and original_file_col in df_group.columns:
                for file_identifier in df_group[original_file_col]:
                    if pd.isna(file_identifier):
                        continue

                    # Clean and normalize the row identifier string
                    row_id_clean = Path(str(file_identifier).strip()).stem.lower()
                    matched_files = []

                    for f in disk_files:
                        f_lower = f.lower()
                        f_stem_lower = Path(f).stem.lower()

                        if accent_name == "indian":
                            if f_stem_lower == row_id_clean or f_stem_lower.startswith(row_id_clean + "_"):
                                matched_files.append(f)
                        else:
                            # Substring match: check if the row identifier is contained inside the file name
                            if row_id_clean in f_lower or f_stem_lower in row_id_clean:
                                matched_files.append(f)

                    # Copy all verified files for this group split
                    for actual_file_name in set(matched_files):
                        src_file_path = accent_raw_dir / actual_file_name
                        dest_file_path = group_output_dir / actual_file_name
                        if not dest_file_path.exists():
                            shutil.copy2(src_file_path, dest_file_path)
                            copied_files_count += 1

            if copied_files_count == 0 and len(df_group) > 0:
                print(
                    f"     [!] Warning: Checked {len(df_group)} rows for '{group_name}' but found NO matching audio files inside '{accent_raw_dir.name}'.")

            # Apply structural renaming map AFTER copying physical files
            if config["rename_map"]:
                df_group = df_group.rename(columns=config["rename_map"])

            # Trim DataFrame down to targeted valid columns only right before saving
            keep_columns = [c for c in config["keep_cols"] if c in df_group.columns]
            df_group = df_group[keep_columns]

            # Save out target tracking CSV sheet
            group_meta_name = f"{accent_name}_{group_name}_metadata.csv"
            df_group.to_csv(group_output_dir / group_meta_name, index=False, encoding='utf-8-sig')

            print(
                f" -> Group [{group_name.upper()}]: Saved {len(df_group)} rows. Copied {copied_files_count} audio tracks.")

    print("\nDataset Splitter Pipeline Executed Successfully.")


if __name__ == "__main__":
    split_dataset()