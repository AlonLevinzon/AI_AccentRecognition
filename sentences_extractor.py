from pydub import AudioSegment
import os
from pathlib import Path
from audio_time_split import *

def cut_audio_to_sentences():

    base_input_path = "accents_samples/accent_archive/"
    base_output_path = "sentences"

    # Define the 6 segments exactly as we planned
    stella_target_segments = [
        "Please call Stella. Ask her to bring these things",
        "with her from the store: Six spoons of fresh snow peas,",
        "five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "We also need a small plastic snake and a big toy frog for the kids.",
        "She can scoop these things into three red bags,",
        "and we will go meet her Wednesday at the train station."
    ]

    if not os.path.exists(base_input_path):
        print(f"Error: Input path {base_input_path} does not exist.")
        return

    # Iterate through all accent sub-folders
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
                # Load audio using pydub
                audio = AudioSegment.from_file(file_path)

                # CRITICAL FIX: Passed the target segments into the alignment function
                sentences_list = get_sentences_time(file_path, stella_target_segments)

                for sentence_index, sentence, start_time, end_time in sentences_list:
                    if not sentence:  # Skip if the sentence text is empty
                        continue

                    # Convert seconds to milliseconds for pydub slicing
                    start_ms = int(start_time * 1000)
                    end_ms = int(end_time * 1000)

                    # Safety check: ensure we don't slice past the actual audio length
                    end_ms = min(len(audio), end_ms)

                    # Slice the audio chunk
                    chunk = audio[start_ms:end_ms]

                    # CRITICAL FIX: Use 'sentence_{index}' instead of the raw text
                    # to avoid illegal character errors (like ':' or '.') in folder/file names.
                    audio_folder = os.path.join(base_output_path, folder_name, f"sentence_{sentence_index}")
                    if not os.path.exists(audio_folder):
                        os.makedirs(audio_folder)

                    # Save the chunk using a clean, safe naming convention
                    chunk_name = os.path.join(audio_folder, f"{file_stem}_part_{sentence_index}.wav")
                    chunk.export(chunk_name, format="wav")

                print(f"Finished splitting {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
if __name__ == "__main__":
    # If you are testing audio_time_split.py directly:
    print("Starting process...")
    cut_audio_to_sentences()
    # print(result)