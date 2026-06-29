import pandas as pd
import os


def create_metadata_excel():
    print("Loading Kaldi files...")
    folder_list = ["train","test"]
    for folder in folder_list:
        folder_path = f"accents_audio/speechocean762/{folder}"
        # 1. Load files mapped by Utterance ID (space or tab separated)
        # File linking each utterance (audio clip) to its corresponding speaker ID
        utt2spk = pd.read_csv(os.path.join(folder_path, "utt2spk"), sep=r'\s+', names=["utt_id", "speaker_id"], dtype=str)

        # File containing the audio file paths
        wav_scp = pd.read_csv(os.path.join(folder_path, "wav.scp"), sep=r'\s+', names=["utt_id", "wav_path"], dtype=str)

        # File containing transcriptions (text)
        # Since transcription contains spaces, we read the line and split only on the first whitespace
        with open(os.path.join(folder_path, "text"), 'r', encoding='utf-8') as f:
            text_lines = [line.strip().split(None, 1) for line in f if line.strip()]
        text_df = pd.DataFrame(text_lines, columns=["utt_id", "transcription"])

        # 2. Load files mapped by Speaker ID
        # File containing speaker gender info
        spk2gender = pd.read_csv(os.path.join(folder_path, "spk2gender"), sep=r'\s+', names=["speaker_id", "gender"],
                                 dtype=str)

        # File containing speaker age info (check if file exists first)
        if os.path.exists(os.path.join(folder_path, "spk2age")):
            spk2age = pd.read_csv(os.path.join(folder_path, "spk2age"), sep=r'\s+', names=["speaker_id", "age"], dtype=str)
        else:
            spk2age = pd.DataFrame(columns=["speaker_id", "age"])

        print("Merging dataframes based on relational keys...")

        # 3. Perform relational merges (Joins)
        # First, merge all utterance-level dataframes using 'utt_id' as the key
        metadata = pd.merge(wav_scp, utt2spk, on="utt_id", how="left")
        metadata = pd.merge(metadata, text_df, on="utt_id", how="left")

        # Second, merge speaker-level metadata together using 'speaker_id' as the key
        speaker_info = pd.merge(spk2gender, spk2age, on="speaker_id", how="outer")

        # Finally, map speaker metadata onto the main utterance dataframe using 'speaker_id'
        final_metadata = pd.merge(metadata, speaker_info, on="speaker_id", how="left")

        # 4. Reorder columns for better readability and structure
        column_order = ["utt_id", "speaker_id", "gender", "age", "wav_path", "transcription"]
        # Ensure we only sort columns that actually exist (in case 'age' was missing)
        column_order = [col for col in column_order if col in final_metadata.columns]
        final_metadata = final_metadata[column_order]


        # 5. Save the perfectly aligned dataset to a structured Excel spreadsheet
        output_filename = f"{folder}_metadata.xlsx"
        print(f"Saving to Excel: {output_filename}")
        final_metadata.to_excel(output_filename, index=False)
        print("Done successfully!")


if __name__ == "__main__":
    # Run the script inside the directory containing your Kaldi files
    create_metadata_excel()