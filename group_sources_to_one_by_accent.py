import os
import shutil
import warnings
import pandas as pd

# Suppress potential openpyxl warnings when reading older Excel formats
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ==========================================
# Configuration and Global Parameters
# ==========================================
BASE_INPUT_DIR = "all_accents_data"  # Parent directory containing all metadata files and subfolders
BASE_OUTPUT_DIR = "audio_by_accent_raw"
TOTAL_SAMPLES_PER_ACCENT = 1200
SEED = 42

# Adjustable ratios for dataset mixing (e.g., 0.70/0.30, can be changed to 0.80/0.20)
PRIMARY_RATIO = 0.70
SECONDARY_RATIO = 0.30

# Flexible mapping lists for columns that represent the same entity
FILE_COLUMNS = ['filename', 'file_name', 'audio_file', 'file', 'wav_path', 'wave_path', 'audio', 'path']
SPEAKER_COLUMNS = ['speaker_id', 'client_id', 'speaker', 'user_id']
TEXT_COLUMNS = ['text', 'transcription', 'sentence', 'transcript']

# Detailed mapping folder names to  match  folder directory
CONFIG = {
    "indian": {
        "primary": {
            "folder": "indian_fleurs_accent",
            "metadata": "indian_fleurs_metadata.csv"  # FIXED: Removed the 't'
        },
        "secondary": {
            "folder": "indian_mozzila_accent",
            "metadata": "indian_mozzila_metadata.csv"
        }
    },
    "chinese": {
        "primary": {
            "folder": "chinese_speechOcean_accent",
            "metadata": "chinese_speechOcean_metadata.xlsx"
        },
        "secondary": {
            "folder": "chinese_ascend_accent",
            "metadata": "chinese_ascend_metadata.csv"
        }
    },
    "nigerian": {
        "primary": {
            "folder": "nigerian_mozzila_accent",
            "metadata": "nigerian_mozzila_metadata.xlsx"
        },
        "secondary": {
            "folder": "nigerian_slr_70_accent",
            "metadata": "nigerian__slr70_metadata.csv"
        }
    },
    "usa": {
        "primary": {
            "folder": "usa_libriSpeech_accent",
            "metadata": "usa_libriSpeech_metadata.xlsx",
            "force_file_column": "file"  # Explicitly force 'file' column to avoid picking up 'audio'
        },
        "secondary": {
            "folder": "usa_mozzila_accent",
            "metadata": "usa_common_voice_metadata.csv"
        }
    }
}


# ==========================================
# Helper Functions
# ==========================================

def load_metadata(parent_dir, metadata_filename, force_file_column=None):
    """
    Loads the designated metadata file (CSV or Excel) located directly in the parent directory,
    filtering and normalizing it to exactly three columns: file_name, speaker_id, and text.
    Additionally cleans the file_name column to strip out any folder prefixes.
    """
    metadata_file = os.path.join(parent_dir, metadata_filename)

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file '{metadata_filename}' not found in: {parent_dir}")

    # Load metadata based on file extension
    if metadata_file.endswith('.csv'):
        df = pd.read_csv(metadata_file)
    else:
        df = pd.read_excel(metadata_file)

    # 1. Resolve File Name Column
    found_file_col = None
    if force_file_column and force_file_column in df.columns:
        found_file_col = force_file_column
    else:
        for col in FILE_COLUMNS:
            if col in df.columns:
                found_file_col = col
                break

    # 2. Resolve Speaker ID Column
    found_speaker_col = None
    for col in SPEAKER_COLUMNS:
        if col in df.columns:
            found_speaker_col = col
            break

    # 3. Resolve Text Column
    found_text_col = None
    for col in TEXT_COLUMNS:
        if col in df.columns:
            found_text_col = col
            break

    # If any of the three mandatory mappings failed, show debug info and raise exception
    if not found_file_col or not found_speaker_col or not found_text_col:
        print(f"DEBUG: Available columns in {metadata_filename}: {list(df.columns)}")
        print(f"DEBUG: Mapped columns -> file: {found_file_col}, speaker: {found_speaker_col}, text: {found_text_col}")
        raise KeyError(
            f"Required columns (file identifier, speaker_id, text) could not be mapped in {metadata_filename}")

    # Filter only the 3 required columns and rename them for uniform output
    df = df[[found_file_col, found_speaker_col, found_text_col]].copy()
    df.rename(columns={
        found_file_col: 'file_name',
        found_speaker_col: 'speaker_id',
        found_text_col: 'text'
    }, inplace=True)

    # Drop rows with missing crucial identifiers
    df.dropna(subset=['file_name', 'speaker_id'], inplace=True)

    # Strip any directory path prefixes (like 'audio_files/') to keep only the pure base filename
    df['file_name'] = df['file_name'].apply(lambda x: os.path.basename(str(x).replace('\\', '/')).strip())

    df['speaker_id'] = df['speaker_id'].astype(str)

    return df


def get_samples_split(df, target_count, excluded_speakers=None):
    """
    Selects samples up to the target_count while preventing Speaker Leakage.
    Splits are performed strictly at the speaker level using a deterministic random seed.
    Can accept a set of excluded_speakers to prevent overlap if we run multiple passes.
    """
    if excluded_speakers is None:
        excluded_speakers = set()

    # Filter out any already used speakers to strictly prevent speaker leakage
    filtered_df = df[~df['speaker_id'].isin(excluded_speakers)]

    # Extract unique speakers and shuffle them deterministically using the set SEED
    unique_speakers = filtered_df['speaker_id'].drop_duplicates().sample(frac=1, random_state=SEED).tolist()

    selected_samples = []
    current_count = 0
    used_speakers = set()

    for speaker in unique_speakers:
        speaker_df = filtered_df[filtered_df['speaker_id'] == speaker]
        speaker_samples_count = len(speaker_df)

        # Guardrail: Prevent excessive overshoot of the sample target (limit to target + 10)
        if current_count + speaker_samples_count > target_count + 10:
            break

        selected_samples.append(speaker_df)
        current_count += speaker_samples_count
        used_speakers.add(speaker)

        # Stop collecting once target count is satisfied
        if current_count >= target_count:
            break

    if not selected_samples:
        return pd.DataFrame(columns=['file_name', 'speaker_id', 'text']), used_speakers

    return pd.concat(selected_samples, ignore_index=True), used_speakers


def find_and_copy_file_flat(src_folder, file_name, dest_path):
    """
    Strictly flat file matching. Checks if the file exists directly inside the source directory.
    Also handles potential extension mismatches (e.g., .mp3 vs .wav) for safety.
    """
    if not os.path.exists(src_folder):
        return False

    base_name = os.path.splitext(file_name)[0].lower()

    # 1. Quick direct match
    direct_path = os.path.join(src_folder, file_name)
    if os.path.exists(direct_path):
        shutil.copy2(direct_path, dest_path)
        return True

    # 2. Flexible match (ignores extension like .mp3/.wav and handles case sensitivity)
    for actual_file in os.listdir(src_folder):
        actual_file_path = os.path.join(src_folder, actual_file)

        # Ensure we are looking at a file (not a folder)
        if os.path.isfile(actual_file_path):
            if os.path.splitext(actual_file)[0].lower() == base_name:
                dest_dir = os.path.dirname(dest_path)
                shutil.copy2(actual_file_path, os.path.join(dest_dir, actual_file))
                return True

    return False


# ==========================================
# Main Processing Pipeline
# ==========================================

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Calculate initial target counts based on the defined ratios
    primary_target = round(TOTAL_SAMPLES_PER_ACCENT * PRIMARY_RATIO)
    secondary_target = round(TOTAL_SAMPLES_PER_ACCENT * SECONDARY_RATIO)

    print(f"--- Starting Audio Mixing Process ---")
    print(f"Target Ratio: {PRIMARY_RATIO * 100}% / {SECONDARY_RATIO * 100}%")
    print(f"Target Samples: {primary_target} (Primary), {secondary_target} (Secondary)")
    print(f"Strict Target Per Accent: {TOTAL_SAMPLES_PER_ACCENT}\n")

    for accent, config in CONFIG.items():
        print(f"Processing accent: {accent.upper()}...")

        accent_output_dir = os.path.join(BASE_OUTPUT_DIR, accent)
        os.makedirs(accent_output_dir, exist_ok=True)

        # Build paths for the source audio folders
        primary_folder = os.path.join(BASE_INPUT_DIR, config['primary']['folder'])
        secondary_folder = os.path.join(BASE_INPUT_DIR, config['secondary']['folder'])

        # 1. Load source metadata datasets from the parent directory (BASE_INPUT_DIR)
        try:
            force_p = config['primary'].get('force_file_column')
            force_s = config['secondary'].get('force_file_column')

            df_primary_all = load_metadata(BASE_INPUT_DIR, config['primary']['metadata'], force_file_column=force_p)
            df_secondary_all = load_metadata(BASE_INPUT_DIR, config['secondary']['metadata'], force_file_column=force_s)
        except Exception as e:
            print(f"Error loading metadata for {accent}: {e}")
            continue

        # 2. First-pass: Attempt standard split according to ratio
        df_primary_selected, primary_speakers = get_samples_split(df_primary_all, primary_target)
        df_secondary_selected, secondary_speakers = get_samples_split(df_secondary_all, secondary_target)

        collected_primary = len(df_primary_selected)
        collected_secondary = len(df_secondary_selected)
        total_collected = collected_primary + collected_secondary

        print(
            f"  - Initial Split: Selected {collected_primary} (Primary) and {collected_secondary} (Secondary). Total: {total_collected}")

        # 3. Dynamic compensation: If total samples are less than target, complete from the other source
        if total_collected < TOTAL_SAMPLES_PER_ACCENT:
            needed_samples = TOTAL_SAMPLES_PER_ACCENT - total_collected
            print(f"  - [Compensation Mode Active] Deficit of {needed_samples} samples detected.")

            # Case A: Primary source had enough, but Secondary was depleted
            if collected_secondary < secondary_target:
                print(
                    f"    Attempting to pull extra {needed_samples} samples from Primary ({config['primary']['metadata']})...")
                df_extra, extra_speakers = get_samples_split(
                    df_primary_all,
                    needed_samples,
                    excluded_speakers=primary_speakers
                )
                if len(df_extra) > 0:
                    df_primary_selected = pd.concat([df_primary_selected, df_extra], ignore_index=True)
                    primary_speakers.update(extra_speakers)
                    print(f"    Successfully added {len(df_extra)} extra samples from Primary source.")

            # Case B: Secondary source had enough, but Primary was depleted
            elif collected_primary < primary_target:
                print(
                    f"    Attempting to pull extra {needed_samples} samples from Secondary ({config['secondary']['metadata']})...")
                df_extra, extra_speakers = get_samples_split(
                    df_secondary_all,
                    needed_samples,
                    excluded_speakers=secondary_speakers
                )
                if len(df_extra) > 0:
                    df_secondary_selected = pd.concat([df_secondary_selected, df_extra], ignore_index=True)
                    secondary_speakers.update(extra_speakers)
                    print(f"    Successfully added {len(df_extra)} extra samples from Secondary source.")

        # Re-evaluate final composition
        df_combined = pd.concat([df_primary_selected, df_secondary_selected], ignore_index=True)
        print(f"  - Final composition: Primary = {len(df_primary_selected)}, Secondary = {len(df_secondary_selected)}")
        print(f"  - Total collected samples for {accent}: {len(df_combined)}")

        # 4. Copy physical audio files from their subfolders and save consolidated metadata
        copied_count = 0
        missing_files_report = []

        # Identify which files came from which source for strict reporting
        primary_filenames = set(df_primary_selected['file_name'])

        for idx, row in df_combined.iterrows():
            file_name = row['file_name']
            dest_path = os.path.join(accent_output_dir, file_name)

            # Record where we expect to find the file
            is_primary = file_name in primary_filenames
            expected_folder = primary_folder if is_primary else secondary_folder

            copied = False
            if is_primary:
                if find_and_copy_file_flat(primary_folder, file_name, dest_path):
                    copied = True
            else:
                if find_and_copy_file_flat(secondary_folder, file_name, dest_path):
                    copied = True

            if copied:
                copied_count += 1
            else:
                # Store missing file info for debug reporting
                missing_files_report.append((file_name, expected_folder))

        # Export the final consolidated metadata CSV for the accent
        metadata_output_path = os.path.join(accent_output_dir, f"{accent}_metadata.csv")
        df_combined.to_csv(metadata_output_path, index=False)

        print(f"  - Successfully copied {copied_count} files to {accent_output_dir}")
        print(f"  - Saved consolidated metadata to {metadata_output_path}")

        # Print detailed report of missing files if any exist
        if missing_files_report:
            print(f"  - [DEBUG WARNING] Could not copy {len(missing_files_report)} files:")
            for i, (missing_file, folder) in enumerate(missing_files_report[:10]):  # Show first 10
                print(f"    {i + 1}. '{missing_file}' was expected in '{folder}' but not found.")
            if len(missing_files_report) > 10:
                print(f"    ... and {len(missing_files_report) - 10} more missing files.")
        print()

    print("--- Process Completed Successfully! ---")


if __name__ == "__main__":
    main()