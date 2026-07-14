import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define directories paths based on project layout
CURRENT_DIR = Path(__file__).resolve().parent
RAW_AUDIO_DIR = CURRENT_DIR / "audio_by_accent_raw"
SENTENCES_DIR = CURRENT_DIR / "sentences_samples"

# Output Directories
FREE_TEXT_OUT_DIR = CURRENT_DIR / "machine_learning_official_free_text_only"
MIXED_OUT_DIR = CURRENT_DIR / "machine_learning_official_mixed"

# Standardized column names for the final metadata
FINAL_COLUMNS = ['filename', 'file_path', 'speaker_id', 'accent', 'text']

# Hardcoded Stella sentences mapped by segment index (0 to 5)
STELLA_SENTENCES = [
    "Please call Stella Ask her to bring these things",
    "with her from the store Six spoons of fresh snow peas",
    "five thick slabs of blue cheese and maybe a snack for her brother Bob",
    "We also need a small plastic snake and a big toy frog for the kids",
    "She can scoop these things into three red bags",
    "and we will go meet her Wednesday at the train station"
]


# =====================================================================
# HELPER FUNCTIONS FOR DIRECT METADATA LOADING & CLEANING
# =====================================================================

def load_exact_metadata(folder_path, accent_name):
    """
    Directly loads the metadata file based on expected fixed names.
    Tries Excel first, then falls back to CSV.
    """
    excel_path = folder_path / f"{accent_name}_metadata.xlsx"
    csv_path = folder_path / f"{accent_name}_metadata.csv"

    try:
        if excel_path.exists():
            df = pd.read_excel(excel_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            print(f" -> Warning: No metadata file found at {excel_path} or {csv_path}")
            return None

        # Standardize columns to lowercase and stripped spaces
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        print(f" -> Error reading metadata for {accent_name}: {e}")
        return None


def get_column_value(df, alternative_keys, row_idx):
    """
    Uses df.at[row_idx, key] to safely query using the actual index label,
    preventing IndexError out-of-bounds crashes after drop_duplicates rows shifts.
    """
    for key in alternative_keys:
        if key in df.columns:
            val = df.at[row_idx, key]
            return val if pd.notna(val) else ""
    return ""


def clean_filename_path(raw_filename):
    """
    Advanced extraction that parses both 'sampling_rate' and 'path'
    dictionary-like strings (e.g., USA dataset formats) safely.
    """
    raw_str = str(raw_filename).strip()

    if "'path':" in raw_str or '"path":' in raw_str:
        raw_str = raw_str.replace("{", "").replace("}", "")
        parts = raw_str.split(",")
        for p in parts:
            if "path" in p:
                raw_str = p.split(":")[-1].strip().strip("'").strip('"')
                break

    if "sampling_rate" in raw_str:
        raw_str = raw_str.replace("{", "").replace("}", "")
        parts = raw_str.split(",")
        for p in parts:
            if "sampling_rate" not in p:
                raw_str = p.strip().strip("'").strip('"')
                break

    return os.path.basename(raw_str)


def clean_text_quotes(text_val):
    """
    Completely sanitizes text strings for phonetic processing.
    Removes quotes, commas, colons, question/exclamation marks, and periods.
    Converts to lowercase to avoid vocabulary splitting mismatches.
    """
    if pd.isna(text_val) or not text_val:
        return ""

    cleaned = str(text_val).lower().strip()

    # Target and strip out all unnecessary non-tonal punctuation markers
    punctuation_to_remove = ['"', "'", ',', ':', '.', '?', '!', ';', '-', '_']
    for char in punctuation_to_remove:
        cleaned = cleaned.replace(char, '')

    # Standardize whitespace distribution patterns
    cleaned = " ".join(cleaned.split())
    return cleaned


def clean_speaker_id(speaker_val):
    """
    Ensures speaker_id is formatted as a standardized clean string.
    """
    if pd.isna(speaker_val) or speaker_val == "":
        return "unknown_speaker"

    if isinstance(speaker_val, float) or (isinstance(speaker_val, str) and '.' in speaker_val):
        try:
            return str(int(float(speaker_val)))
        except ValueError:
            pass
    return str(speaker_val).strip()


def sanitize_accent_label(accent_val, folder_accent):
    """
    Enforces centralized accent label constraints (e.g., maps 'hindi' to 'indian').
    """
    if not accent_val or pd.isna(accent_val):
        return folder_accent.lower()

    cleaned = str(accent_val).strip().lower()
    if cleaned == 'hindi':
        return 'indian'
    return cleaned


def sample_dataset_safely(df, target_limit, mode='files'):
    """
    Flexible grouping sampler that limits dataset sizes based on row/file targets
    or absolute distinct unique speaker caps. Includes empty DataFrame safety guards.
    """
    if df.empty:
        return df

    unique_speakers = df['speaker_id'].unique()
    np.random.shuffle(unique_speakers)

    selected_rows = []
    current_count = 0

    for speaker_idx, speaker in enumerate(unique_speakers, 1):
        speaker_data = df[df['speaker_id'] == speaker]

        if mode == 'speakers':
            if speaker_idx <= target_limit:
                selected_rows.append(speaker_data)
            else:
                break
        else:
            if current_count + len(speaker_data) <= target_limit:
                selected_rows.append(speaker_data)
                current_count += len(speaker_data)
            else:
                if not selected_rows:
                    selected_rows.append(speaker_data)
                    current_count += len(speaker_data)
                continue

    if selected_rows:
        return pd.concat(selected_rows, ignore_index=True)
    return pd.DataFrame(columns=df.columns)


def save_pure_csv_without_quotes(df, output_path):
    """
    Writes final metadata lists into plain text files manually.
    Guarantees absolute exclusion of wrapping quotation marks and escape slashes.
    """
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(FINAL_COLUMNS) + "\n")
        if df.empty:
            return
        for _, row in df.iterrows():
            line_parts = [
                str(row['filename']),
                str(row['file_path']),
                str(row['speaker_id']),
                str(row['accent']),
                str(row['text'])
            ]
            f.write(",".join(line_parts) + "\n")


# =====================================================================
# CORE DATA PROCESSING PIPELINES
# =====================================================================

def build_free_text_only_metadata():
    """
    Pipeline for free_text_only: Extracts from audio_by_accent_raw.
    Targets 1200 samples for Train (80%), making total target = 1500 samples per accent.
    """
    print("\n--- Building Free Text Only Massive Metadata ---")
    all_accents_data = []
    accents = ['chinese', 'indian', 'nigerian', 'usa']
    target_total = 1500

    for accent in accents:
        accent_dir = RAW_AUDIO_DIR / accent
        if not accent_dir.exists():
            print(f" -> Directory missing: {accent_dir}")
            continue

        df_raw = load_exact_metadata(accent_dir, accent)
        if df_raw is None:
            continue

        accent_records = []
        for idx, row in df_raw.iterrows():
            raw_filename = get_column_value(df_raw, ['filename', 'file_name', 'audio', 'file', 'wav_path'], idx)
            if not raw_filename:
                continue

            pure_filename = clean_filename_path(raw_filename)
            full_audio_path = accent_dir / pure_filename

            speaker_raw = get_column_value(df_raw, ['speakerid', 'speaker_id', 'client_id', 'id'], idx)
            speaker_id = clean_speaker_id(speaker_raw)

            accent_raw = get_column_value(df_raw, ['accent'], idx)
            accent_label = sanitize_accent_label(accent_raw, accent)

            text_raw = get_column_value(df_raw, ['text', 'transcription', 'sentence'], idx)
            text_str = clean_text_quotes(text_raw)

            accent_records.append({
                'filename': clean_filename_path(raw_filename),
                'file_path': str(full_audio_path),
                'speaker_id': speaker_id,
                'accent': accent_label,
                'text': text_str,
                'source_type': 'raw'
            })

        df_accent = pd.DataFrame(accent_records)
        df_accent_capped = sample_dataset_safely(df_accent, target_total, mode='files')
        print(f" -> Accent '{accent}': Retained {len(df_accent_capped)} samples safely grouped by speaker.")
        all_accents_data.append(df_accent_capped)

    if all_accents_data:
        df_final = pd.concat(all_accents_data, ignore_index=True)
        os.makedirs(FREE_TEXT_OUT_DIR, exist_ok=True)
        save_pure_csv_without_quotes(df_final, FREE_TEXT_OUT_DIR / "massive_metadata_free_text.csv")
        return df_final
    return pd.DataFrame()


def build_mixed_metadata():
    """
    Pipeline for mixed mode: Merges audio_by_accent_raw (capped at 900 files)
    and sentences_samples (capped strictly at a maximum of 60 unique speakers per accent).
    """
    print("\n--- Building Mixed Massive Metadata ---")
    all_records = []
    accents = ['chinese', 'indian', 'nigerian', 'usa']

    # Part 1: Process Raw Audio Folder (Capped at 900 total files per accent)
    for accent in accents:
        accent_dir = RAW_AUDIO_DIR / accent
        if not accent_dir.exists():
            continue

        df_raw = load_exact_metadata(accent_dir, accent)
        if df_raw is None:
            continue

        accent_records = []
        for idx, row in df_raw.iterrows():
            raw_filename = get_column_value(df_raw, ['filename', 'file_name', 'audio', 'file', 'wav_path'], idx)
            if not raw_filename:
                continue

            pure_filename = clean_filename_path(raw_filename)
            full_path = accent_dir / pure_filename

            speaker_raw = get_column_value(df_raw, ['speakerid', 'speaker_id', 'client_id', 'id'], idx)
            speaker_id = clean_speaker_id(speaker_raw)

            accent_raw = get_column_value(df_raw, ['accent'], idx)
            accent_label = sanitize_accent_label(accent_raw, accent)

            text_raw = get_column_value(df_raw, ['text', 'transcription', 'sentence'], idx)
            text_str = clean_text_quotes(text_raw)

            accent_records.append({
                'filename': clean_filename_path(raw_filename),
                'file_path': str(full_path),
                'speaker_id': speaker_id,
                'accent': accent_label,
                'text': text_str,
                'source_type': 'raw'
            })

        df_accent = pd.DataFrame(accent_records)
        df_accent_capped = sample_dataset_safely(df_accent, 900, mode='files')
        all_records.append(df_accent_capped)
        print(f" -> Raw Accent '{accent}': Sampled {len(df_accent_capped)} files for Mixed Master List.")

    # Part 2: Process Sentences Samples (Archive) Folder (Capped at 60 UNIQUE SPEAKERS per accent)
    for accent in accents:
        accent_dir = SENTENCES_DIR / accent
        if not accent_dir.exists():
            print(f" -> Archive directory missing for accent: {accent}, skipping archive split.")
            continue

        df_raw = load_exact_metadata(accent_dir, accent)
        if df_raw is None:
            continue

        speaker_col_name = None
        for key in ['speakerid', 'speaker_id', 'client_id', 'id']:
            if key in df_raw.columns:
                speaker_col_name = key
                break

        df_unique_speakers = df_raw.copy()
        if speaker_col_name:
            df_unique_speakers['temp_spk'] = df_unique_speakers[speaker_col_name].apply(clean_speaker_id)
            df_unique_speakers = df_unique_speakers.drop_duplicates(subset=['temp_spk'])

        sentence_records = []
        for idx, row in df_unique_speakers.iterrows():
            raw_filename = get_column_value(df_unique_speakers, ['filename', 'file_name', 'audio', 'file', 'wav_path'],
                                            idx)
            if not raw_filename:
                continue

            pure_filename = clean_filename_path(raw_filename)
            speaker_id = clean_speaker_id(
                get_column_value(df_unique_speakers, ['speakerid', 'speaker_id', 'client_id', 'id'], idx))

            accent_raw = get_column_value(df_unique_speakers, ['accent'], idx)
            accent_label = sanitize_accent_label(accent_raw, accent)

            # Extract pure name without extension to build 'mandarin9_part_X.wav' format dynamically
            base_name, _ = os.path.splitext(pure_filename)

            for i in range(6):
                # RECONSTRUCT ARCHIVE FILENAME ACCORDING TO ACTUAL DISK PATTERNS: 'mandarin9_part_0.wav'
                archive_filename = f"{base_name}_part_{i}.wav"

                path_with_folder = accent_dir / f"sentence_{i}" / archive_filename
                path_direct = accent_dir / archive_filename

                valid_path = None
                if path_with_folder.exists():
                    valid_path = path_with_folder
                elif path_direct.exists():
                    valid_path = path_direct

                # Append records only if a valid physical file path exists on disk
                if valid_path is not None:
                    final_text = clean_text_quotes(STELLA_SENTENCES[i])

                    sentence_records.append({
                        'filename': archive_filename,  # Save the accurate structural file name
                        'file_path': str(valid_path),  # Save the absolute valid file path
                        'speaker_id': speaker_id,
                        'accent': accent_label,
                        'text': final_text,
                        'source_type': 'archive'
                    })

        df_sentence = pd.DataFrame(sentence_records)

        if df_sentence.empty:
            print(f" -> Sentences Accent '{accent}': 0 valid physical files found. Skipping archive pool sampling.")
            continue

        df_sentence_capped = sample_dataset_safely(df_sentence, target_limit=60, mode='speakers')

        retained_speakers = len(df_sentence_capped['speaker_id'].unique()) if not df_sentence_capped.empty else 0
        print(
            f" -> Sentences Accent '{accent}': Capped/Retained {retained_speakers} unique speakers ({len(df_sentence_capped)} valid existing files extracted).")

        all_records.append(df_sentence_capped)

    if all_records:
        df_final = pd.concat(all_records, ignore_index=True)
        os.makedirs(MIXED_OUT_DIR, exist_ok=True)
        save_pure_csv_without_quotes(df_final, MIXED_OUT_DIR / "massive_metadata_mixed.csv")
        return df_final
    return pd.DataFrame()


# =====================================================================
# SPEAKER LEAKAGE VERIFICATION DETECTOR
# =====================================================================

def verify_zero_speaker_leakage(train_ids, val_ids, test_ids):
    """
    Mathematically validates that there is absolute zero intersection
    of speaker identities between Train, Validation, and Test indices.
    """
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    leak_train_val = set_train.intersection(set_val)
    leak_train_test = set_train.intersection(set_test)
    leak_val_test = set_val.intersection(set_test)

    print("   ↳ Leakage Status ->", end=" ")
    if leak_train_val or leak_train_test or leak_val_test:
        print("[CRITICAL WARNING] SPEAKER LEAKAGE DETECTED!")
        return False
    else:
        print(
            f"[SUCCESS] Zero leakage. Unique Speakers -> Train: {len(set_train)} | Val: {len(set_val)} | Test: {len(set_test)}")
        return True


# =====================================================================
# UNIFIED NESTED K-FOLD SPLITTING ENGINE (80/10/10)
# =====================================================================

def run_nested_group_kfold(df):
    """
    Unifies cross-validation using an interleaved GroupKFold structure.
    Solves speaker leakage by grouping overlapping identities from raw and archive pools together.

    Returns a list of dictionaries, where each dict represents a fold combination:
    {
        'outer_fold': int,
        'inner_fold': int,
        'train_df': pd.DataFrame,
        'val_df': pd.DataFrame,
        'test_df': pd.DataFrame
    }
    """
    if df.empty:
        print("Dataframe is empty. Cannot perform K-Fold calculations.")
        return []

    if 'source_type' not in df.columns:
        df = df.copy()
        df['source_type'] = 'raw'

    df = df.reset_index(drop=True)
    X = df[FINAL_COLUMNS].values
    groups = df['speaker_id'].values

    outer_k = 10
    inner_k = 9

    outer_gkf = GroupKFold(n_splits=outer_k)

    # Store all calculated splits for downstream ML pipelines
    all_folds_data = []

    print(f"\n========================================================")
    print(f"Executing Unified Nested Group K-Fold (80% / 10% / 10%)")
    print(f"========================================================")

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_gkf.split(X, groups=groups), 1):
        df_test = df.iloc[test_idx].copy()
        df_train_val = df.iloc[train_val_idx]

        X_inner = df_train_val[FINAL_COLUMNS].values
        groups_inner = df_train_val['speaker_id'].values

        inner_gkf = GroupKFold(n_splits=inner_k)

        for inner_fold_idx, (train_rel_idx, val_rel_idx) in enumerate(inner_gkf.split(X_inner, groups=groups_inner), 1):
            train_idx = train_val_idx[train_rel_idx]
            val_idx = train_val_idx[val_rel_idx]

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            r_train = len(df_train[df_train['source_type'] == 'raw'])
            r_val = len(df_val[df_val['source_type'] == 'raw'])
            r_test = len(df_test[df_test['source_type'] == 'raw'])

            a_train = len(df_train[df_train['source_type'] == 'archive'])
            a_val = len(df_val[df_val['source_type'] == 'archive'])
            a_test = len(df_test[df_test['source_type'] == 'archive'])

            print(f"\n>>> [OUTER FOLD {fold_idx}/{outer_k}] -> [INNER GRIDSEARCH FOLD {inner_fold_idx}/{inner_k}]")
            print(f"  Raw Samples     -> Train: {r_train} | Val: {r_val} | Test: {r_test}")
            print(f"  Archive Samples -> Train: {a_train} | Val: {a_val} | Test: {a_test}")
            print(f"  Total Combined  -> Train: {len(df_train)} ({len(df_train) / len(df) * 100:.1f}%) | "
                  f"Val: {len(df_val)} ({len(df_val) / len(df) * 100:.1f}%) | "
                  f"Test: {len(df_test)} ({len(df_test) / len(df) * 100:.1f}%)")

            # Validate structural speaker isolation across sets
            is_valid = verify_zero_speaker_leakage(df_train['speaker_id'], df_val['speaker_id'], df_test['speaker_id'])

            if is_valid:
                # Append finalized split dataframes for feature extraction mapping
                all_folds_data.append({
                    'outer_fold': fold_idx,
                    'inner_fold': inner_fold_idx,
                    'train_df': df_train,
                    'val_df': df_val,
                    'test_df': df_test
                })

    return all_folds_data


def run_nested_group_kfold_generator(df):
    """
    Generator version of the GroupKFold engine.
    Yields tuple splits sequentially for the main execution loop.
    """
    X = df[['filename', 'file_path', 'speaker_id', 'accent', 'text']].values
    groups = df['speaker_id'].values
    outer_k, inner_k = 10, 9

    outer_gkf = GroupKFold(n_splits=outer_k)

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_gkf.split(X, groups=groups), 1):
        df_test = df.iloc[test_idx]
        df_train_val = df.iloc[train_val_idx]

        X_inner = df_train_val[['filename', 'file_path', 'speaker_id', 'accent', 'text']].values
        groups_inner = df_train_val['speaker_id'].values
        inner_gkf = GroupKFold(n_splits=inner_k)

        for train_rel_idx, val_rel_idx in inner_gkf.split(X_inner, groups=groups_inner):
            train_idx = train_val_idx[train_rel_idx]
            val_idx = train_val_idx[val_rel_idx]

            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

            # Yield indices matrices control chunks directly to the new main script
            yield fold_idx, df_train, df_val, df_test
            break  # Match gridsearch setup sequence
# =====================================================================
# MAIN RUNNER PROGRAM ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    # Dictionary to store the processed splits for downstream ML tasks
    datasets_splits = {}

    # Task 1: Generate and validate the Free-Text Only master sheet layout
    df_free_text = build_free_text_only_metadata()
    if not df_free_text.empty:
        print("\nMassive Free-Text Master Matrix successfully established.")
        # All breaks removed; gathers all 90 cross-validation split combinations
        datasets_splits['free_text'] = run_nested_group_kfold(df_free_text)

    print("\n" + "=" * 60 + "\n")

    # Task 2: Generate and validate the Mixed master dataset sheet layout
    df_mixed = build_mixed_metadata()
    if not df_mixed.empty:
        print("\nMassive Mixed Master Matrix successfully established.")
        datasets_splits['mixed'] = run_nested_group_kfold(df_mixed)

    # Sanity check to confirm splits are correctly allocated in memory
    if 'free_text' in datasets_splits and datasets_splits['free_text']:
        print(f"\n[INFO] Generated {len(datasets_splits['free_text'])} total valid splits for Free-Text.")
    if 'mixed' in datasets_splits and datasets_splits['mixed']:
        print(f"[INFO] Generated {len(datasets_splits['mixed'])} total valid splits for Mixed.")