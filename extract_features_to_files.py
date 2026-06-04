from audio_features import *
import os
import pandas as pd

def import_vector_to_file():
    """
       Iterates through language folders,uses function create_features_vector to extracts linear-scale Mel-Spectrograms from
       audio folders, then exports them a csv file .
       """
    #create columns name for all features except MFCC
    feature_names = [
        'pitch_mean', 'pitch_std', 'average_volume', 'volume_variance',
        'volume_max', 'volume_min', 'mean_centroid', 'mean_rolloff',
        'mean_zcr', 'total_duration', 'phonation_time', 'phonation_ratio',
        'syllable_rate', 'vowel_consonant_ratio', 'f1', 'f2'
    ]

    # adding MFCC and delta 1 & 2 columns names
    feature_names += [f'mfcc_{i}' for i in range(1, 14)]
    feature_names += [f'delta_{i}' for i in range(1, 14)]
    feature_names += [f'delta2_{i}' for i in range(1, 14)]
    all_columns = ['file_name', 'speaker_id'] + feature_names + ['accent']

    dataset_rows=[]

    base_dir = "accents_samples/accent_archive/"

    for folder_name in os.listdir(base_dir):
        accent_folder_path = os.path.join(base_dir, folder_name)

        # Skip files, process directories only
        if not os.path.isdir(accent_folder_path):
            continue

        accent_label = folder_name
        # Ensure we are processing an actual directory
        if os.path.isdir(accent_folder_path):
            print(f"Processing accent group: {accent_label}")

        # iterate audio files on each language folder
        for file_name in os.listdir(accent_folder_path):
            # Validate standard audio file extensions
            if not file_name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                continue
            audio_file_path = os.path.join(accent_folder_path, file_name)

            # Derive a unique speaker ID from the filename (example 'arabic45')
            speaker_id = os.path.splitext(file_name)[0]

            try:
                #extract features for audio using create_features_vector below (55 features)
                features_vector = create_features_vector(audio_file_path)
                row_data = [file_name, speaker_id] + list(features_vector) + [accent_label]
                dataset_rows.append(row_data)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

        #convering the columns and rows to data structure with labeled axes (rows and columns) from Pandas Python Library
        df = pd.DataFrame(dataset_rows, columns=all_columns)
        #creating the csv file
        df.to_csv('accent_features_dataset.csv', index=False)
        print("csv file successfully created")

        # # Dropping unnumerical features for the Machine Learning Model
        # X = df.drop(columns=['file_name', 'speaker_id', 'accent'])
        # y = df['accent']  # Now 'y' contains the automatically extracted labels

def create_features_vector(file_path):
    #load audio with  standardized sampling rate (16kHz)
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y) #TODO: DO I need it # Strip leading/trailing structural silence

    #basic features

    # 1. Extract acoustic components
    rms, average_volume, volume_variance, volume_max, volume_min = extract_rms_features(y)
    zcr, mean_zcr = extract_zero_crossing_rate(y)
    f0, pitch_mean, pitch_std = extract_pitch_features(y, sr)

    # 2. Extract spectral shape and brightness features
    spec_centroid, mean_centroid = extract_spectral_centroid(y, sr)
    spec_rolloff, mean_rolloff = extract_spectral_rolloff(y, sr)

    # 3. Extract physical time and fluency features
    total_duration, phonation_time, phonation_ratio = extract_duration_and_phonation(y,sr,f0)

    #4. Extract advanced linguistic features
    syllable_rate = extract_advanced_syllable_rate(y, sr, phonation_time)
    vowel_consonant_ratio = extract_vowel_consonant_ratio(f0, zcr, rms)
    f1, f2 = extract_formant_frequencies(y, sr)

    # 5. Extract dynamic MFCC features -This captures the static vocal filter alongside its 1st (Delta) and 2nd (Delta-Delta) time derivatives
    mfcc, delta_mfcc, delta2_mfcc, mfcc_dynamic = extract_mfcc_features(y, sr)

    #the vector contains 16 scalars
    features_vector = np.array([
        pitch_mean,             # Pitch global center (Hz)
        pitch_std,              # Intonation volatility / Tonal marker (Hz)
        average_volume,         # Signal root mean square intensity average
        volume_variance,        # Syllable stress rhythm / Energy variance
        volume_max,             # Peak articulatory vocal stress
        volume_min,             # Floor energy limits
        mean_centroid,          # Spectral center of mass / Perceived brightness (Hz)
        mean_rolloff,           # Spectral asymmetry / 85% energy cutoff boundary (Hz)
        mean_zcr,               # Average zero crossing rate / Fricative noise proxy
        total_duration,         # Complete phrase file length (Seconds)
        phonation_time,         # True voiced duration excluding structural gaps (Seconds)
        phonation_ratio,        # Speech fluency index / Hesitation marker
        syllable_rate,          # Rhythm pacing score / Syllable count per speech second
        vowel_consonant_ratio,  # Phonetic balance profile
        f1,                     # Tongue height indicator / Open-closed jaw resonance (Hz)
        f2                      # Tongue advancement indicator / Front-back resonance (Hz)
    ])
    #Concatenate the 16 base scalars with the 39-dimensional MFCC dynamic vector
    # Total vector shape will be exactly 55 dimensions
    complete_feature_vector = np.concatenate([features_vector, mfcc_dynamic])
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