import os
import pandas as pd
import numpy as np
import librosa
from audio_features import *
from Dynamic_features import *
from phonetic_features import *

def import_vector_to_file():
    """
    Iterates through language folders, maps safe folder names (e.g., sentence_0) back
    to their original English text, extracts all acoustic and phonetic features, and saves to CSV.
    """
    # Define the exact target text segments mapped by their structural index (from sentences_extractor.py)
    STELLA_TEXT_MAP = {
        "sentence_0": "Please call Stella. Ask her to bring these things",
        "sentence_1": "with her from the store: Six spoons of fresh snow peas,",
        "sentence_2": "five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "sentence_3": "We also need a small plastic snake and a big toy frog for the kids.",
        "sentence_4": "She can scoop these things into three red bags,",
        "sentence_5": "and we will go meet her Wednesday at the train station."
    }

    # Create column names for the 16 baseline scalar features
    base_feature_names = [
        'pitch_mean', 'pitch_std', 'average_volume', 'volume_variance',
        'volume_max', 'volume_min', 'mean_centroid', 'mean_rolloff',
        'mean_zcr', 'total_duration', 'phonation_time', 'phonation_ratio',
        'syllable_rate', 'vowel_consonant_ratio', 'f1', 'f2'
    ]

    # Add 13 MFCC and their 1st & 2nd time derivatives (39 features)
    mfcc_feature_names = []
    mfcc_feature_names += [f'mfcc_{i}' for i in range(1, 14)]
    mfcc_feature_names += [f'delta_{i}' for i in range(1, 14)]
    mfcc_feature_names += [f'delta2_{i}' for i in range(1, 14)]

    # Add 9 advanced dynamic biomarkers from Dynamic_features.py (Praat/Parselmouth)
    advanced_acoustic_names = [
        'praat_mean_f0', 'praat_std_f0', 'pitch_velocity_std',
        'jitter', 'shimmer', 'mean_F1', 'mean_F2', 'mean_F3', 'F2_F1_ratio'
    ]

    # Add 5 structural phonetic vector features (Theoretical linguistics)
    phonetic_feature_names = [
        'phn_voiced', 'phn_fricative', 'phn_plosive', 'phn_nasal', 'phn_long_vowel'
    ]

    # Master columns layout (Total length: 2 metadata columns + 69 features + 1 label = 72 columns)
    all_columns = (
            ['file_name', 'speaker_id'] +
            base_feature_names +
            mfcc_feature_names +
            advanced_acoustic_names +
            phonetic_feature_names +
            ['accent']
    )

    dataset_rows = []
    base_dir = "sentences_samples"

    for folder_name in os.listdir(base_dir):
        accent_folder_path = os.path.join(base_dir, folder_name)

        # Skip files, process directories only
        if not os.path.isdir(accent_folder_path):
            continue

        accent_label = folder_name
        print(f"Processing accent group: {accent_label}")

        # Iterate through sentence folders (e.g., "sentence_0", "sentence_1")
        for sentence_folder in os.listdir(accent_folder_path):
            sentence_file_path = os.path.join(accent_folder_path, sentence_folder)

            if not os.path.isdir(sentence_file_path):
                continue

            # Retrieve the true English text corresponding to this safe folder name
            # Fallback to empty string if folder name doesn't match the expected dictionary keys
            ground_truth_text = STELLA_TEXT_MAP.get(sentence_folder, "")

            # Iterate through the actual audio files inside the sentence partition folder
            for file_name in os.listdir(sentence_file_path):
                # Validate standard audio file extensions
                if not file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    continue

                audio_file_path = os.path.join(accent_folder_path, sentence_folder, file_name)

                # Derive a unique speaker ID from the filename (e.g., 'arabic45')
                speaker_id = os.path.splitext(file_name)[0]

                try:
                    # Pass the recovered English text into the feature extraction engine
                    features_vector = create_features_vector(audio_file_path, ground_truth_text=ground_truth_text)

                    # Store row data using the directory name (sentence_folder) as the file identifier
                    row_data = [sentence_folder, speaker_id] + list(features_vector) + [accent_label]
                    dataset_rows.append(row_data)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

    # Convert structural rows and columns into a Pandas DataFrame
    df = pd.DataFrame(dataset_rows, columns=all_columns)

    # Export the engineered feature matrix to a CSV file
    df.to_csv('accent_features_dataset.csv', index=False)
    print("CSV dataset successfully created with safe text mapping!")


def create_features_vector(file_path, ground_truth_text=None):#we put NONE so it can work only on acoustic features if no text is known
    """
    Extracts 16 base scalars, 39 MFCC dynamic components, 9 advanced physiological features,
    and 5 normalized phonetic attributes, returning a unified flat vector.
    """
    # Load audio with standardized sampling rate (16kHz)
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y)  # Strip leading/trailing structural silence

    # 1. Extract baseline acoustic components
    rms, average_volume, volume_variance, volume_max, volume_min = extract_rms_features(y)
    zcr, mean_zcr = extract_zero_crossing_rate(y)
    f0, pitch_mean, pitch_std = extract_pitch_features(y, sr)

    # 2. Extract spectral shape and brightness features
    spec_centroid, mean_centroid = extract_spectral_centroid(y, sr)
    spec_rolloff, mean_rolloff = extract_spectral_rolloff(y, sr)

    # 3. Extract physical time and fluency features
    total_duration, phonation_time, phonation_ratio = extract_duration_and_phonation(y, sr, f0)

    # 4. Extract advanced linguistic vocal-tract features
    syllable_rate = extract_advanced_syllable_rate(y, sr, phonation_time)
    vowel_consonant_ratio = extract_vowel_consonant_ratio(f0, zcr, rms)
    f1, f2 = extract_formant_frequencies(y, sr)

    # 5. Extract dynamic MFCC features (Static vocal filter + 1st & 2nd time derivatives)
    mfcc, delta_mfcc, delta2_mfcc, mfcc_dynamic = extract_mfcc_features(y, sr)

    # Compile the 16 baseline scalar features
    base_scalars = np.array([
        pitch_mean, pitch_std, average_volume, volume_variance,
        volume_max, volume_min, mean_centroid, mean_rolloff,
        mean_zcr, total_duration, phonation_time, phonation_ratio,
        syllable_rate, vowel_consonant_ratio, f1, f2
    ])

    # 6. Extract the 9 advanced dynamic biomarkers from Dynamic_features.py (Praat backend)
    # Using try/except in case Praat fails on extremely short or corrupted audio frames
    try:
        advanced_acoustic_features = extract_advanced_acoustic_features(file_path)
    except Exception:
        advanced_acoustic_features = np.zeros(9)

    # 7. Extract the 5 normalized theoretical phonetic features using the hybrid pipeline
    # Disabling internal ASR (allow_asr=False) since the text context is provided directly from the directory structure
    phonetic_features = get_phonetic_features(file_path, ground_truth_text=ground_truth_text, allow_asr=False)
    phonetic_features = np.array(phonetic_features)

    # -------------------------------------------------------------------------
    # MULTI-MODAL FEATURE FUSION (Concatenation)
    # Total Matrix Shape calculation: 16 (base) + 39 (mfcc) + 9 (praat) + 5 (phonetic) = 69 Dimensions
    # -------------------------------------------------------------------------
    complete_feature_vector = np.concatenate([
        base_scalars,
        mfcc_dynamic,
        advanced_acoustic_features,
        phonetic_features
    ])

    return complete_feature_vector

def create_spectrogram_folder():
    """
    Iterates through language folders, extracts linear-scale Mel-Spectrograms from
    audio folders, then exports them as clean PNG images formatted specifically for CNN training.
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
    import_vector_to_file()

if __name__ == "__main__":
    main()