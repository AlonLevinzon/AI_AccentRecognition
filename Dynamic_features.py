import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


def extract_advanced_acoustic_features(audio_path):
    """
    Extracts physiological and dynamic voice biomarkers inspired by Premananth et al. (2025).
    Returns a 1D numpy array containing 9 engineered scalar features.
    """
    features = {}

    # Load audio into a high-fidelity Praat Sound Object for precise acoustic mapping
    sound = parselmouth.Sound(audio_path)

    # ---------------------------------------------------------
    # 1. Pitch (F0) Extraction & Dynamics (Premananth et al., 2025)
    # ---------------------------------------------------------
    # Extract fundamental frequency (F0) using autocorrelation (75Hz-600Hz)
    pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
    pitch_values = pitch.selected_array['frequency']

    # Filter out unvoiced frames (F0=0) to isolate harmonic speech data
    voiced_pitch = pitch_values[pitch_values > 0]

    if len(voiced_pitch) > 0:
        mean_f0 = np.mean(voiced_pitch)
        std_f0 = np.std(voiced_pitch)

        # Normalize F0 by the speaker's mean to eliminate gender and biological bias
        normalized_pitch = voiced_pitch / mean_f0

        features['mean_f0'] = mean_f0
        features['std_f0'] = std_f0
        # Compute the velocity (1st derivative) of normalized pitch to capture prosodic rhythm
        features['pitch_velocity_std'] = np.std(np.diff(normalized_pitch))
    else:
        features['mean_f0'] = 0
        features['std_f0'] = 0
        features['pitch_velocity_std'] = 0

    # ---------------------------------------------------------
    # 2. Glottal Micro-Instability: Jitter & Shimmer
    # ---------------------------------------------------------
    # Generate glottal pulses (PointProcess) to track vocal fold closure timestamps
    point_process = call([sound, pitch], "To PointProcess (cc)")

    # Extract Jitter (local) to measure cycle-to-cycle frequency variations
    jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    # Extract Shimmer (local) to measure cycle-to-cycle amplitude/loudness variations
    shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    features['jitter'] = jitter if not np.isnan(jitter) else 0
    features['shimmer'] = shimmer if not np.isnan(shimmer) else 0

    # ---------------------------------------------------------
    # 3. Vocal Tract Resonance: Formants (F1, F2, F3)
    # ---------------------------------------------------------
    # Apply Linear Predictive Coding (LPC Burg) to map vocal tract resonances up to 5500Hz
    formant = call(sound, "To Formant (burg)", 0.0, 5.0, 5500.0, 0.025, 50.0)

    num_frames = call(formant, "Get number of frames")
    f1_list, f2_list, f3_list = [], [], []

    # Iterate dynamically through time-frames to sample F1, F2, and F3 center frequencies
    for frame in range(1, num_frames + 1):
        t = call(formant, "Get time from frame number", frame)
        f1 = call(formant, "GetValue at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "GetValue at time", 2, t, "Hertz", "Linear")
        f3 = call(formant, "GetValue at time", 3, t, "Hertz", "Linear")

        if not np.isnan(f1): f1_list.append(f1)
        if not np.isnan(f2): f2_list.append(f2)
        if not np.isnan(f3): f3_list.append(f3)

    features['mean_F1'] = np.mean(f1_list) if f1_list else 0
    features['mean_F2'] = np.mean(f2_list) if f2_list else 0
    features['mean_F3'] = np.mean(f3_list) if f3_list else 0

    # Compute the F2/F1 ratio to geometrically map the speaker's native vowel space
    features['F2_F1_ratio'] = features['mean_F2'] / features['mean_F1'] if features['mean_F1'] > 0 else 0

    # Compile the 9 engineered biomarkers into a fixed-size 1D feature vector
    feature_vector = [
        features['mean_f0'], features['std_f0'], features['pitch_velocity_std'],
        features['jitter'], features['shimmer'],
        features['mean_F1'], features['mean_F2'], features['mean_F3'],
        features['F2_F1_ratio']
    ]

    return np.array(feature_vector)


# ---------------------------------------------------------
# 4. Feature Fusion Integration
# ---------------------------------------------------------
if __name__ == "__main__":
    example_audio = "path_to_your_audio.wav"
    existing_55_features = np.random.rand(55)

    try:
        new_features = extract_advanced_acoustic_features(example_audio)
        # Concatenate baseline MFCC features with new physiological features (Total: 64 features)
        extended_acoustic_vector = np.concatenate((existing_55_features, new_features))
        print("Extended Vector Length:", len(extended_acoustic_vector))
    except Exception as e:
        print(f"Error: {e}")