import os
import shutil
import pandas as pd

DATA_GROUPS = ['train', 'test']
# Note: Adjusted to point to the directory containing the 'WAVE' folder based on image paths
AUDIO_BASE_DIR = 'accents_audio/speechocean762'
# Updated template extension to .xlsx
METADATA_PATH_TEMPLATE = 'accents_audio/speechocean762/{group}_metadata.xlsx'
TARGET_DIR = './audio_by_accent/chinese'

for group in DATA_GROUPS:
    print(f"--- Processing group: {group} ---")

    # 1. Load the Excel metadata file
    metadata_path = METADATA_PATH_TEMPLATE.format(group=group)
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found for {group} at {metadata_path}")
        continue

    # Changed from pd.read_csv to pd.read_excel
    df = pd.read_excel(metadata_path)

    # Define target path for the current group (e.g., ./audio_by_accent/chinese/train)
    dest_group_path = os.path.join(TARGET_DIR, group)
    os.makedirs(dest_group_path, exist_ok=True)

    # List to track rows of successfully copied files for the new metadata
    copied_rows = []

    # 2. Group the metadata by speaker_id to easily take the first 10 samples per speaker
    grouped = df.groupby('speaker_id')

    for speaker_id, speaker_df in grouped:
        # Take up to the first 10 rows/samples for this speaker
        first_10_samples = speaker_df.head(10)

        for index, row in first_10_samples.iterrows():
            # Use 'wav_path' directly from the metadata
            relative_wav_path = row['wav_path']

            # Construct absolute source path
            src_file_path = os.path.join(AUDIO_BASE_DIR, relative_wav_path)

            # Extract just the filename to save it in the target directory
            filename = os.path.basename(relative_wav_path)
            dest_file_path = os.path.join(dest_group_path, filename)

            # 3. Copy file and update the metadata tracking list
            if os.path.exists(src_file_path):
                shutil.copy2(src_file_path, dest_file_path)

                # Update the row's wav_path to reflect its new flat location inside the group folder
                new_row = row.copy()
                new_row['wav_path'] = filename
                copied_rows.append(new_row)
            else:
                print(f"File missing: {src_file_path}")

    # 4. Save the filtered metadata as an Excel file (.xlsx) for this specific group
    if copied_rows:
        new_metadata_df = pd.DataFrame(copied_rows)
        new_metadata_df.reset_index(drop=True, inplace=True)

        # Updated output format to Excel (.xlsx) using to_excel
        output_excel_path = os.path.join(dest_group_path, f"chinese_{group}_metadata.xlsx")
        new_metadata_df.to_excel(output_excel_path, index=False)
        print(f"Successfully saved updated Excel metadata at: {output_excel_path}")

print("\nDone! Selected samples and Excel metadata files have been cleanly separated.")