import pandas as pd
import os


def run_dataset_sanity_checks():
    """
    Performs rigorous validation on Train, Validation, and Test feature sheets
    to detect Speaker Leakage, row duplications, and data anomalies.
    """
    BASE_DIR = "official_audio_free_text_only"

    print("\n=============================================")
    print("RUNNING ANTI-DATA-LEAKAGE SANITY CHECKS")
    print("=============================================")

    # 1. Load the original RAW features (before scaling) to check real metadata strings
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(BASE_DIR, "validation_features.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test_features.csv"))

    # ---------------------------------------------------------
    # CHECK 1: SPEAKER LEAKAGE DETECTION (The Most Critical Audio Bug)
    # ---------------------------------------------------------
    train_speakers = set(train_df['speaker_id'].dropna().unique())
    val_speakers = set(val_df['speaker_id'].dropna().unique())
    test_speakers = set(test_df['speaker_id'].dropna().unique())

    print(
        f"Unique Speakers Found:\n - Train: {len(train_speakers)}\n - Val: {len(val_speakers)}\n - Test: {len(test_speakers)}")

    # Check intersection between splits
    train_test_leakage = train_speakers.intersection(test_speakers)
    train_val_leakage = train_speakers.intersection(val_speakers)
    val_test_leakage = val_speakers.intersection(test_speakers)

    has_leakage = False

    if len(train_test_leakage) > 0:
        print(
            f"🚨 CRITICAL BUG: Speaker Leakage detected between TRAIN and TEST! Overlapping speakers: {train_test_leakage}")
        has_leakage = True
    else:
        print("✅ SUCCESS: Zero Speaker Leakage detected between Train and Test sets.")

    if len(train_val_leakage) > 0:
        print(
            f"⚠️ WARNING: Speaker Leakage detected between TRAIN and VALIDATION! Overlapping speakers: {train_val_leakage}")
        has_leakage = True
    else:
        print("✅ SUCCESS: Zero Speaker Leakage detected between Train and Validation sets.")

    # ---------------------------------------------------------
    # CHECK 2: EXACT ROW DUPLICATION DETECTION
    # ---------------------------------------------------------
    # Combine datasets temporarily to check for identical audio feature rows across splits
    combined_features = pd.concat([
        train_df.drop(columns=['filename', 'speaker_id', 'text', 'accent'], errors='ignore'),
        val_df.drop(columns=['filename', 'speaker_id', 'text', 'accent'], errors='ignore'),
        test_df.drop(columns=['filename', 'speaker_id', 'text', 'accent'], errors='ignore')
    ])

    total_rows = len(combined_features)
    duplicate_rows_count = combined_features.duplicated().sum()

    if duplicate_rows_count > 0:
        print(
            f" ANOMALY DETECTED: Found {duplicate_rows_count} identical feature rows out of {total_rows} total rows across the dataset!")
        has_leakage = True
    else:
        print("✅ SUCCESS: No identical row duplications found across feature matrices.")

    # ---------------------------------------------------------
    # CHECK 3: FILENAME DUPLICATION DETECTION
    # ---------------------------------------------------------
    all_filenames = pd.concat([train_df['filename'], val_df['filename'], test_df['filename']])
    duplicate_files = all_filenames.duplicated().sum()
    if duplicate_files > 0:
        print(f"🚨 WARNING: Found {duplicate_files} duplicate filenames shared across splits!")
        has_leakage = True
    else:
        print("✅ SUCCESS: All filenames across splits are unique.")

    print("=============================================\n")
    return has_leakage