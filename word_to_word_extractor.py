from pydub import AudioSegment
import os
from pathlib import Path
from audio_time_split import *


def cut_audio_to_words():
    base_input_path = "accents_samples/accent_archive/"
    base_output_path = "word_to_word"

    if not os.path.exists(base_input_path):
        print(f"Error: Input path {base_input_path} does not exist.")
        return

    #iterate through all sub_folders in the folder
    for folder_name in os.listdir(base_input_path):
        input_folder_path = os.path.join(base_input_path, folder_name)

        if not os.path.isdir(input_folder_path):
            continue


        for i,file_name in enumerate(os.listdir(input_folder_path)):
            if not file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue

            file_path = os.path.join(input_folder_path, file_name)
            file_stem = Path(file_name).stem

            print(f"Processing: {file_name}...")

            try:
                #audio loading
                audio = AudioSegment.from_file(file_path)

                # Get the list of words in paragraph
                words_list = get_words_time(file_path)

                for word_index,word, start_time, end_time in words_list:
                    if not word:  # Skip if the word is empty
                        continue

                    # Convert seconds to milliseconds for pydub slicing
                    start_ms = int(start_time * 1000)
                    end_ms = int(end_time * 1000)
                    #safety_end_ms = min(len(audio), end_ms + 300)
                    # Slice the audio chunk
                    chunk = audio[start_ms:end_ms]

                    # create folder for every word audio if it doesn't exist
                    audio_folder = os.path.join(base_output_path, folder_name, f"{word}")
                    if not os.path.exists(audio_folder):
                        os.makedirs(audio_folder)

                    # saving the split
                    chunk_name = os.path.join(audio_folder, f"{word}_{file_stem}_{i}.wav")
                    chunk.export(chunk_name, format="wav")

                print(f"Finished splitting {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
if __name__ == "__main__":
    # If you are testing audio_time_split.py directly:
    print("Starting process...")
    cut_audio_to_words()
    # print(result)