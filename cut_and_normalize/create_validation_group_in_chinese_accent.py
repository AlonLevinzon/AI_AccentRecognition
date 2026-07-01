import os
import pandas as pd
from pathlib import Path
import shutil


def copy_group_files(df_partition, file_col, disk_files_map, source_audio_dir, destination_dir):
    copied_count = 0
    for file_identifier in df_partition[file_col]:
        if pd.isna(file_identifier):
            continue
        lookup_key = str(file_identifier).strip().lower()
        matched_name = disk_files_map.get(lookup_key, disk_files_map.get(Path(lookup_key).stem.lower()))

        if matched_name:
            src_path = source_audio_dir / matched_name
            dest_path = destination_dir / matched_name
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
            copied_count += 1
    return copied_count


def create_new_chinese_split():
    # Detect project root directory safely
    current_dir = Path.cwd()
    if current_dir.name == "cut_and_normalize":
        root_dir = current_dir.parent
    else:
        root_dir = current_dir

    # FIXED PATHS: Based on your exact location guidelines
    raw_base_path = root_dir / "audio_by_accent_raw"
    chinese_source_audio_dir = raw_base_path / "chinese" / "train"  # Audio inside audio_by_accent_raw/chinese/train

    target_meta_name = "chinese_train_metadata.xlsx"
    uploaded_meta_path = raw_base_path / target_meta_name  # Metadata inside audio_by_accent_raw/

    # Verify input metadata exists
    if not uploaded_meta_path.exists():
        print(f"Error: Target Excel metadata file not found at {uploaded_meta_path.resolve()}")
        print(f"Please ensure '{target_meta_name}' is placed directly inside 'audio_by_accent_raw' folder.")
        return

    print(f"Loading metadata sheet: {uploaded_meta_path.name}...")

    # -------------------------------------------------------------
    # שונה לקריאת אקסל (שינוי השורה המבוקשת)
    df = pd.read_excel(uploaded_meta_path)
    # -------------------------------------------------------------

    # Hardcoded exact columns from your screenshot
    file_col = "wav_path"
    speaker_col = "speaker_id"
    text_col = "transcription"

    if speaker_col not in df.columns or file_col not in df.columns:
        print(f"Error: Expected columns not found in Excel. Found: {df.columns.tolist()}")
        return

    # Step 1: Scan from bottom to top to collect unique validation speakers until files >= 268
    print("Isolating validation speakers from the bottom of the dataset rows...")
    unique_speakers_reverse = []
    seen = set()
    for spk in reversed(df[speaker_col]):
        if pd.notna(spk) and spk not in seen:
            seen.add(spk)
            unique_speakers_reverse.append(spk)

    val_speakers = set()
    accumulated_files = 0
    target_val_count = 268

    for spk in unique_speakers_reverse:
        speaker_rows_count = len(df[df[speaker_col] == spk])
        val_speakers.add(spk)
        accumulated_files += speaker_rows_count

        # Stop as soon as we safely cross or hit 268 to prevent split speakers
        if accumulated_files >= target_val_count:
            break

    print(
        f"Collected {len(val_speakers)} unique validation speakers yielding exactly {accumulated_files} audio tracks.")

    # Step 2: Split DataFrame into clean, mutually exclusive subsets
    df_val = df[df[speaker_col].isin(val_speakers)].copy()
    df_train = df[~df[speaker_col].isin(val_speakers)].copy()

    # Step 3: Create target directories under root/new_chinese/
    output_base_path = root_dir / "new_chinese"
    train_output_dir = output_base_path / "train"
    val_output_dir = output_base_path / "validation"

    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)

    # Cache disk files from the correct source directory for quick lookup
    disk_files_map = {}
    if chinese_source_audio_dir.exists():
        for f in os.listdir(chinese_source_audio_dir):
            if os.path.isfile(chinese_source_audio_dir / f):
                disk_files_map[f.lower()] = f
                disk_files_map[Path(f).name.lower()] = f
                disk_files_map[Path(f).stem.lower()] = f
    else:
        print(f"Error: Raw Chinese source audio directory missing at {chinese_source_audio_dir.resolve()}")
        return

    print(f"\nCopying Train audio files to {train_output_dir.name}...")
    train_copied = copy_group_files(df_train, file_col, disk_files_map, chinese_source_audio_dir, train_output_dir)

    print(f"Copying Validation audio files to {val_output_dir.name}...")
    val_copied = copy_group_files(df_val, file_col, disk_files_map, chinese_source_audio_dir, val_output_dir)

    # Step 4: Standardize and sanitize the final output schemas
    rename_rules = {file_col: "audio", speaker_col: "client_id", text_col: "sentence"}
    df_train = df_train.rename(columns=rename_rules)
    df_val = df_val.rename(columns=rename_rules)

    # Enforce Chinese accent tag
    df_train["accent"] = "Chinese"
    df_val["accent"] = "Chinese"

    keep_cols = ["audio", "client_id", "sentence", "accent"]
    df_train = df_train[keep_cols]
    df_val = df_val[keep_cols]

    # Save out the clean tracker sheets (נשאר שמירה כ-CSV כפי שהיה במקור)
    df_train.to_csv(train_output_dir / "chinese_train_metadata.csv", index=False, encoding='utf-8-sig')
    df_val.to_csv(val_output_dir / "chinese_validation_metadata.csv", index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(" NEW CHINESE DATASET GENERATION SUCCESSFUL")
    print("=" * 60)
    print(f" -> Base Directory Location: {output_base_path.resolve()}")
    print(f" -> New Train: {len(df_train)} rows registry. Copied {train_copied} audio files.")
    print(f" -> New Validation: {len(df_val)} rows registry. Copied {val_copied} audio files.")
    print("=" * 60)


if __name__ == "__main__":
    create_new_chinese_split()