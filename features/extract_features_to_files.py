import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from features.audio_features import *
from features.Dynamic_features import *
from features.phonetic_features import *
import traceback


def find_file_in_dir(base_dir, filename):
    """
    Helper function to locate a file within base_dir or any of its subdirectories.
    Returns the full path if found, else None.
    """
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def import_vector_to_file(metadata_csv_path, audio_base_dir="../official_audio_mixed"):
    """
    Reads a metadata CSV file, iterates through its rows, locates the corresponding
    audio files in audio_base_dir, extracts acoustic/phonetic features using the
    provided text, and exports a unified feature matrix.
    """
    # 1. Load the metadata file
    if not os.path.exists(metadata_csv_path):
        print(f"Error: Metadata file not found at {metadata_csv_path}")
        return

    print(f"Loading metadata from {metadata_csv_path}...")
    metadata_df = pd.read_csv(metadata_csv_path)

    # 2. Define column names for the final feature file
    base_feature_names = [
        'pitch_mean', 'pitch_std', 'average_volume', 'volume_variance',
        'volume_max', 'volume_min', 'mean_centroid', 'mean_rolloff',
        'mean_zcr', 'total_duration', 'phonation_time', 'phonation_ratio',
        'syllable_rate', 'vowel_consonant_ratio', 'f1', 'f2'
    ]

    mfcc_feature_names = []
    mfcc_feature_names += [f'mfcc_{i}' for i in range(1, 14)]
    mfcc_feature_names += [f'delta_{i}' for i in range(1, 14)]
    mfcc_feature_names += [f'delta2_{i}' for i in range(1, 14)]

    advanced_acoustic_names = [
        'praat_mean_f0', 'praat_std_f0', 'pitch_velocity_std',
        'jitter', 'shimmer', 'mean_F1', 'mean_F2', 'mean_F3', 'F2_F1_ratio'
    ]

    phonetic_feature_names = [
        'phn_voiced', 'phn_fricative', 'phn_plosive', 'phn_nasal', 'phn_long_vowel'
    ]

    # Layout of columns in the final exported DataFrame
    all_columns = (
            ['filename', 'speaker_id', 'text'] +
            base_feature_names +
            mfcc_feature_names +
            advanced_acoustic_names +
            phonetic_feature_names +
            ['accent']
    )

    dataset_rows = []

    # 3. Iterate through each row in the metadata file for feature extraction
    print(f"Starting feature extraction for {len(metadata_df)} files...")

    for idx, row in metadata_df.iterrows():
        filename = row['filename']

        # Removed the temporary archive '_part' restriction to process all files natively
        speaker_id = row['speaker_id']
        accent = row['accent']
        ground_truth_text = row['text']

        # Locate the audio file path within the base directory (including subdirectories)
        audio_file_path = find_file_in_dir(audio_base_dir, filename)

        if audio_file_path is None:
            continue

        print(f"[{idx + 1}/{len(metadata_df)}] Processing: {filename} ({accent})")

        try:
            # Pass the file path and corresponding ground truth text to the extraction engine
            features_vector = create_features_vector(audio_file_path, ground_truth_text=ground_truth_text)

            # Construct the dataset row: initial metadata + extracted features + accent label at the end
            row_data = [filename, speaker_id, ground_truth_text] + list(features_vector) + [accent]
            dataset_rows.append(row_data)

        except Exception as e:
            # Cleaned exception handling printing detailed stack trace without emojis
            print(f"\nCRITICAL ERROR on file {filename}!")
            print(f"Error Message: {e}")
            print("--- Full Stack Trace ---")
            traceback.print_exc()
            print("------------------------\n")

    # 4. Convert rows into a DataFrame and export to CSV
    if dataset_rows:
        df_result = pd.DataFrame(dataset_rows, columns=all_columns)
        output_filename = 'accent_features_dataset.csv'
        df_result.to_csv(output_filename, index=False)
        print(f"\nSUCCESS: CSV dataset successfully created with {len(df_result)} rows!")
        print(f"Saved as: '{output_filename}'")
    else:
        print("\nFailure: No features were extracted. Check your audio files or metadata.")


def create_features_vector(file_path, ground_truth_text=None):
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
    ], dtype=float)

    # 6. Extract the 9 advanced dynamic biomarkers from Dynamic_features.py (Praat backend)
    try:
        advanced_acoustic_features = extract_advanced_acoustic_features(file_path)
        advanced_acoustic_features = np.array(advanced_acoustic_features, dtype=float).flatten()
    except Exception as e:
        print(f"Praat Extraction Failed on {file_path}. Reason: {e}")
        advanced_acoustic_features = np.zeros(9, dtype=float)

    # 7. Extract and defend phonetic features against scalar float returns
    try:
        phonetic_features = get_phonetic_features(file_path, ground_truth_text=ground_truth_text, allow_asr=False)
        # If the returned object is a float, scalar or non-iterable, force it to 5 zeroes
        if phonetic_features is None or not hasattr(phonetic_features, "__iter__") or isinstance(phonetic_features,
                                                                                                 (float, int)):
            phonetic_features = np.zeros(5, dtype=float)
        else:
            phonetic_features = np.array(phonetic_features, dtype=float).flatten()
            if len(phonetic_features) != 5:
                phonetic_features = np.zeros(5, dtype=float)
    except Exception:
        phonetic_features = np.zeros(5, dtype=float)

    # Ensure mfcc_dynamic is flat and float
    mfcc_dynamic = np.array(mfcc_dynamic, dtype=float).flatten()

    # FUSION
    complete_feature_vector = np.concatenate([
        base_scalars,
        mfcc_dynamic,
        advanced_acoustic_features,
        phonetic_features
    ])

    return complete_feature_vector


def create_spectrogram_folder():
    base_input_path = "../accents_samples/accent_archive/"
    base_output_path = "../spectrograms"

    for folder_name in os.listdir(base_input_path):
        input_folder_path = os.path.join(base_input_path, folder_name)
        if not os.path.isdir(input_folder_path):
            continue

        spectrogram_folder = os.path.join(base_output_path, folder_name)
        os.makedirs(spectrogram_folder, exist_ok=True)

        for file_name in os.listdir(input_folder_path):
            if not file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue

            input_file_path = os.path.join(input_folder_path, file_name)
            y, sr = librosa.load(input_file_path, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

            raw_file_name = os.path.splitext(file_name)[0]
            output_file_path = os.path.join(spectrogram_folder, f"{raw_file_name}.png")

            plt.figure(figsize=(6, 4))
            librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis=None, y_axis=None)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, format='png', dpi=300)
            plt.close()


def main():
    metadata_file = "metadata.csv"
    import_vector_to_file(metadata_csv_path=metadata_file)


if __name__ == "__main__":
    main()