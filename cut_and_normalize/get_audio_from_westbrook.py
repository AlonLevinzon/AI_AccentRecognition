import os
import shutil
import pandas as pd

#     # =====================================================================
#     # option 1 :mixing train test and validation data (split later)
#     # =================================================
# --- PATH CONFIGURATIONS (Update these paths to match your system) ---
DATA_GROUPS = ['train', 'test', 'validation']
BASE_SOURCE_DIR = '../accents_audio/westbrook-English_Accent_DataSet'  # Folder containing train, test, and validation subfolders
METADATA_PATH_TEMPLATE = 'accents_audio/westbrook-English_Accent_DataSet/metadata_{group}.csv'  # Template path for each group's metadata file
TARGET_DIR = '../audio_by_accent'  # Destination folder for the organized accents

# Mapping accent IDs to their respective target folder names
ACCENT_MAPPING = {
    9: 'spanish',
    19: 'indian',
    22: 'american'
}

# Dictionary to hold accumulated metadata DataFrames for each accent
accent_metadata_buffers = {accent_id: [] for accent_id in ACCENT_MAPPING.keys()}

# --- CREATE TARGET DIRECTORIES ---
for accent_name in ACCENT_MAPPING.values():
    os.makedirs(os.path.join(TARGET_DIR, accent_name), exist_ok=True)

# --- ITERATE THROUGH EACH DATA GROUP ---
for group in DATA_GROUPS:
    print(f"--- Processing group: {group} ---")

    # 1. Load the metadata file for the current group
    metadata_path = METADATA_PATH_TEMPLATE.format(group=group)
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found for {group} at {metadata_path}")
        continue

    df = pd.read_csv(metadata_path)

    # 2. Filter rows that match the target accent IDs (9, 19, 22)
    filtered_df = df[df['accent'].isin(ACCENT_MAPPING.keys())].copy()

    # Track which group this data originally belonged to
    filtered_df['original_group'] = group

    # Define the specific source folder for the current group
    source_folder = os.path.join(BASE_SOURCE_DIR, group)

    # 3. Iterate through filtered rows and copy files
    for index, row in filtered_df.iterrows():
        filename = row['file_name']
        accent_id = int(row['accent'])
        accent_folder_name = ACCENT_MAPPING[accent_id]

        # Construct full source and destination paths
        src_file_path = os.path.join(source_folder, filename)
        dest_file_path = os.path.join(TARGET_DIR, accent_folder_name, filename)

        # Copy the file only if it actually exists in the source folder
        if os.path.exists(src_file_path):
            shutil.copy2(src_file_path, dest_file_path)
            # Append the successfully copied file's row to our metadata buffer
            accent_metadata_buffers[accent_id].append(row)
        else:
            print(f"File missing: {src_file_path}")

# --- SAVE METADATA FOR EACH ACCENT FOLDER ---
print("\n--- Saving metadata per accent folder ---")
for accent_id, rows in accent_metadata_buffers.items():
    accent_folder_name = ACCENT_MAPPING[accent_id]

    if rows:
        # Convert the list of rows back into a single clean DataFrame
        accent_df = pd.DataFrame(rows)

        # Reset index to make it look organized
        accent_df.reset_index(drop=True, inplace=True)

        # Construct the output CSV path inside the accent's folder
        output_csv_path = os.path.join(TARGET_DIR, accent_folder_name, f"{accent_folder_name}_metadata.csv")

        # Save the metadata file
        accent_df.to_csv(output_csv_path, index=False)
        print(f"Saved metadata for {accent_folder_name} at: {output_csv_path}")
    else:
        print(f"No files were processed for accent: {accent_folder_name}")

print("\nDone! All files and their respective metadata have been organized.")


#     # =====================================================================
#     # option 2 :with keeping the split to train test and validation
#     # =================================================
# import os
# import shutil
# import pandas as pd
#
# # --- PATH CONFIGURATIONS (Update these paths to match your system) ---
# DATA_GROUPS = ['train', 'test', 'validation']
# BASE_SOURCE_DIR = './path/to/your/audio_groups'  # Folder containing train, test, and validation subfolders
# METADATA_PATH_TEMPLATE = './path/to/metadata_{group}.csv'  # Template path for each group's metadata file
# TARGET_DIR = './path/to/output_accents'  # Destination folder for the organized accents
#
# # Mapping accent IDs to their respective target folder names
# ACCENT_MAPPING = {
#     9: 'spanish',
#     19: 'indian',
#     22: 'american'
# }
#
# # --- ITERATE THROUGH EACH DATA GROUP ---
# for group in DATA_GROUPS:
#     print(f"--- Processing group: {group} ---")
#
#     # 1. Load the metadata file for the current group
#     metadata_path = METADATA_PATH_TEMPLATE.format(group=group)
#     if not os.path.exists(metadata_path):
#         print(f"Warning: Metadata file not found for {group} at {metadata_path}")
#         continue
#
#     df = pd.read_csv(metadata_path)
#
#     # 2. Filter rows that match the target accent IDs (9, 19, 22)
#     filtered_df = df[df['accent'].isin(ACCENT_MAPPING.keys())]
#
#     # Define the specific source folder for the current group
#     source_folder = os.path.join(BASE_SOURCE_DIR, group)
#
#     # Dictionary to hold rows for metadata saving per accent within this specific group
#     group_metadata_buffers = {accent_id: [] for accent_id in ACCENT_MAPPING.keys()}
#
#     # 3. Iterate through filtered rows and copy files
#     for index, row in filtered_df.iterrows():
#         filename = row['file_name']
#         accent_id = int(row['accent'])
#         accent_folder_name = ACCENT_MAPPING[accent_id]
#
#         # Construct the target directory path for this specific accent and group
#         # Example: ./output_accents/spanish/train/
#         accent_group_dir = os.path.join(TARGET_DIR, accent_folder_name, group)
#         os.makedirs(accent_group_dir, exist_ok=True)
#
#         # Construct full source and destination paths
#         src_file_path = os.path.join(source_folder, filename)
#         dest_file_path = os.path.join(accent_group_dir, filename)
#
#         # Copy the file only if it actually exists in the source folder
#         if os.path.exists(src_file_path):
#             shutil.copy2(src_file_path, dest_file_path)
#             # Store the row for this specific accent's group metadata
#             group_metadata_buffers[accent_id].append(row)
#         else:
#             print(f"File missing: {src_file_path}")
#
#     # 4. Save metadata for each accent within the current group folder
#     for accent_id, rows in group_metadata_buffers.items():
#         accent_folder_name = ACCENT_MAPPING[accent_id]
#         if rows:
#             accent_group_df = pd.DataFrame(rows)
#             accent_group_df.reset_index(drop=True, inplace=True)
#
#             # Save path example: ./output_accents/spanish/train/spanish_train_metadata.csv
#             output_csv_path = os.path.join(TARGET_DIR, accent_folder_name, group,
#                                            f"{accent_folder_name}_{group}_metadata.csv")
#             accent_group_df.to_csv(output_csv_path, index=False)
#             print(f"  -> Saved metadata for {accent_folder_name} ({group})")
#
# print("\nDone! All folders and structured metadata files have been created cleanly.")