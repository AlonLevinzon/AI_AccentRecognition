import os
import sys
import pandas as pd
from pathlib import Path

# Configure local working directory paths
WORKING_DIR = Path.cwd()
PROJECT_DATA_DIR = Path.cwd()

if str(PROJECT_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DATA_DIR))

# Import required pipeline functions
from K_fold import (
    build_free_text_only_metadata,
    build_mixed_metadata,
    build_archive_only_metadata
)
from features.extract_features_to_files import import_vector_to_file

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
# Choose dataset subset mode: "free_text", "mixed", or "archive_only"
RUN_MODE = "free_text"

EXTRACTED_FILENAME = f"master_extracted_features_{RUN_MODE}.csv"
EXTRACTED_FEATURES_MASTER_CSV = WORKING_DIR / EXTRACTED_FILENAME


def extract_features_for_selected_dataset(run_mode: str):
    """
    Standalone subroutine to extract and cache acoustic features for a specified dataset mode.
    Resolves normalized audio file paths dynamically, handles missing files gracefully,
    and saves the extracted feature set into a master CSV file.
    """
    print(f"====================================================")
    print(f"🚀 STARTING STANDALONE FEATURE EXTRACTION ({run_mode.upper()})")
    print(f"====================================================")

    # Step 1: Locate or build metadata for the selected mode
    metadata_file_name = f"massive_metadata_{run_mode}.csv"
    possible_paths = [
        PROJECT_DATA_DIR / f"machine_learning_official_{run_mode}_only" / metadata_file_name,
        PROJECT_DATA_DIR / f"machine_learning_official_{run_mode}" / metadata_file_name,
        PROJECT_DATA_DIR / metadata_file_name
    ]

    df_master = None
    for meta_path in possible_paths:
        if meta_path.exists():
            print(f"📂 [LOAD SUCCESS]: Found existing metadata directly at: {meta_path}")
            df_master = pd.read_csv(meta_path)
            break

    if df_master is None:
        print("⚠️ Metadata CSV not found, triggering builder function...")
        if run_mode == "free_text":
            df_master = build_free_text_only_metadata()
        elif run_mode == "mixed":
            df_master = build_mixed_metadata()
        elif run_mode == "archive_only":
            df_master = build_archive_only_metadata()

    if df_master is None or df_master.empty:
        sys.exit(f"❌ Error: Could not obtain master metadata for mode '{run_mode}'. Extraction aborted.")

    print(f"\n⚡⚡⚡ Extracting acoustic features for ALL {len(df_master)} files in '{run_mode}' dataset...")

    # Step 2: Copy metadata and dynamically resolve normalized audio paths
    df_master_ml = df_master.copy()
    raw_normalized_dir = PROJECT_DATA_DIR / "audio_by_accent_raw_normalized"
    sentences_normalized_dir = PROJECT_DATA_DIR / "sentences_samples_normalized"

    missing_normalized_count = 0
    total_files = len(df_master_ml)

    def resolve_normalized_path(row):
        nonlocal missing_normalized_count

        accent_label = str(row['accent'])
        file_name = str(row['filename'])
        original_path = str(row['file_path'])

        if row.get('source_type') == 'archive' or 'sentence_' in original_path or 'sentences_samples' in original_path:
            part_match = [p for p in Path(original_path).parts if 'sentence_' in p]
            if part_match:
                normalized_path = sentences_normalized_dir / accent_label / part_match[0] / file_name
            else:
                normalized_path = sentences_normalized_dir / accent_label / file_name
        else:
            normalized_path = raw_normalized_dir / accent_label / file_name

        if normalized_path.exists():
            return str(normalized_path)
        else:
            missing_normalized_count += 1
            print(f"⚠️ [NORMALIZATION MISSING]: Normalized file not found for '{file_name}', falling back to original path: {original_path}")
            return original_path

    df_master_ml['file_path'] = df_master_ml.apply(resolve_normalized_path, axis=1)

    print(f"\n🔍 [PATH RESOLUTION SUMMARY]:")
    print(f"   • Total Audio Files Checked: {total_files}")
    print(f"   • Successfully Resolved Normalized Paths: {total_files - missing_normalized_count}")
    print(f"   • Fallback to Original Paths (Missing Normalized): {missing_normalized_count}\n")

    # Step 3: Write temporary pipeline CSV and invoke extraction routine
    temp_master_meta = WORKING_DIR / "temp_master_metadata_pipeline.csv"
    df_master_ml.to_csv(temp_master_meta, index=False, encoding='utf-8-sig')

    audio_base_dir = raw_normalized_dir
    import_vector_to_file(str(temp_master_meta), str(audio_base_dir))

    default_output = WORKING_DIR / "accent_features_dataset.csv"
    if default_output.exists():
        if EXTRACTED_FEATURES_MASTER_CSV.exists():
            os.remove(EXTRACTED_FEATURES_MASTER_CSV)
        os.rename(default_output, EXTRACTED_FEATURES_MASTER_CSV)
        if temp_master_meta.exists():
            os.remove(temp_master_meta)
        print(f"──► [SUCCESS]: Feature dataset created successfully and cached at -> {EXTRACTED_FEATURES_MASTER_CSV}")
    else:
        raise FileNotFoundError("Feature extraction subroutine failed.")

    return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)


if __name__ == "__main__":
    extract_features_for_selected_dataset(RUN_MODE)