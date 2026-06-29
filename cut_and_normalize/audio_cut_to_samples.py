from pydub import AudioSegment
import os
from pathlib import Path

def cut_audio():
    base_input_path = "../accents_samples/accent_archive/"
    base_output_path = "all_samples"
    segment_duration_ms = 3000

    #go through all files in the folder
    for folder_name in os.listdir(base_input_path):
        input_folder_path = os.path.join(base_input_path, folder_name)

        if not os.path.isdir(input_folder_path):
            continue


        for file_name in os.listdir(input_folder_path):
            if not file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue

            file_path = os.path.join(input_folder_path, file_name)
            file_stem = Path(file_name).stem

            print(f"Processing: {file_name}...")

            try:
                #audio loading
                audio = AudioSegment.from_file(file_path)

                #create folder for every audio if it doesn't exist
                audio_folder = os.path.join(base_output_path, folder_name, f"{file_stem}_samples")
                if not os.path.exists(audio_folder):
                    os.makedirs(audio_folder)

                # split the audio to 3 seconds
                for i, start_time in enumerate(range(0, len(audio), segment_duration_ms)):
                    end_time = start_time + segment_duration_ms
                    chunk = audio[start_time:end_time]

                    if len(chunk) < segment_duration_ms:
                        continue

                    # saving the split
                    chunk_name = os.path.join(audio_folder, f"sample_{i:03d}.wav")
                    chunk.export(chunk_name, format="wav")

                print(f"Finished splitting {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
# def main():
#     cut_audio()
#
# if __name__ == "__main__":
#     main()