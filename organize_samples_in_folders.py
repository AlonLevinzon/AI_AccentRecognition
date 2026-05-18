import os
import librosa
import numpy as np
import shutil
from pathlib import Path


def get_data_from_audio():
    base_input_path = "all_samples"
    base_output_path = "all_samples_data"
    # moving across all lang folders
    for folder_name in os.listdir(base_input_path):

        #creating lang_folder
        lag_output_folder = os.path.join(base_output_path, f"{folder_name}_data")
        if not os.path.exists(lag_output_folder):
            os.makedirs(lag_output_folder)

        #moving across all lang_samples folders
        lag_input_folder = os.path.join(base_input_path, f"{folder_name}")
        for i, audio_folder in enumerate(os.listdir(lag_input_folder), start=1):
            #moving across all sample
            audio_sample_folder=os.path.join(lag_input_folder, audio_folder)
            for audio_file in os.listdir(audio_sample_folder):
                #creating folder for every sample i
                sample_i_folder = os.path.join(lag_output_folder, f"{audio_file}_data")
                if not os.path.exists(sample_i_folder):
                    os.makedirs(sample_i_folder)

                #copy the samples to the output_folder
                audio_to_copy = os.path.join(audio_sample_folder, audio_file)
                shutil.copy2(audio_to_copy,sample_i_folder)
                old_name = os.path.join(sample_i_folder, audio_file)
                new_name = Path(old_name).with_stem(f"{Path(audio_file).stem}_{i}")
                os.rename(old_name,new_name)




def main():
    get_data_from_audio()

if __name__ == "__main__":
    main()