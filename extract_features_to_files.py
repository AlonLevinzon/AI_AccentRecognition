import pandas as pd
import numpy as np
from audio_features import *
import os
from pathlib import Path

# def import_vector_to_file():
#     base_input_path = "accents_samples/accent_archive/"
#
#     # file load
#     y, sr = librosa.load(audio_path, sr=None)


def create_spectrogram_folder():
    """
    Iterates through language folders, extracts linear-scale Mel-Spectrograms from
    audio samples, removes visual artifacts (axes, labels, borders), and exports
    them as clean PNG images formatted specifically for CNN training.
    """
    base_input_path = "accents_samples/accent_archive/"
    base_output_path = "spectrograms"

    # Iterate language folders inside the base folder
    for folder_name in os.listdir(base_input_path):
        input_folder_path = os.path.join(base_input_path, folder_name)

        # Skip files, process directories only
        if not os.path.isdir(input_folder_path):
            continue

        # Create output folder
        spectrogram_folder = os.path.join(base_output_path, folder_name)
        os.makedirs(spectrogram_folder, exist_ok=True)

        # iterate audio files on each language folder
        for file_name in os.listdir(input_folder_path):
            # Validate standard audio file extensions
            if not file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue

            input_file_path = os.path.join(input_folder_path, file_name)

            # Load audio with forced unified sampling rate (16kHz) for CNN consistency
            y, sr = librosa.load(input_file_path, sr=16000)

            # Extract Mel-Spectrogram with explicit standard parameters matching the research paper
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
            )

            # Isolate the raw file name without its audio extension (e.g., 'speaker1')
            raw_file_name = os.path.splitext(file_name)[0]
            output_file_path = os.path.join(spectrogram_folder, f"{raw_file_name}.png")

            # Initialize a clean canvas for visual plotting
            plt.figure(figsize=(6, 4))

            # Render the spectrogram matrix without displaying time or frequency ticks/axes
            librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis=None, y_axis=None)

            # Turn off all axis lines and labels entirely
            plt.axis('off')

            # Stretch the visual representation to fill up the entire canvas space
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

            # Export canvas to disk as PNG to keep fine-grained variance intact without lossy artifacts
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, format='png', dpi=300)

            # Close the figure instance immediately to prevent RAM accumulation over massive dataset iterations
            plt.close()

def main():
    create_spectrogram_folder()

if __name__ == "__main__":
    main()