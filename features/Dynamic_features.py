import numpy as np
import parselmouth


def extract_advanced_acoustic_features(audio_path):
    """
    Extracts physiological and dynamic voice biomarkers inspired by Premananth et al. (2025).
    Guarantees all outputs are standard Python floats to prevent numpy dimension mismatch errors.
    """
    features = {}

    # Load audio directly into a high-fidelity Praat Sound Object
    sound = parselmouth.Sound(str(audio_path))

    # ---------------------------------------------------------
    # 1. Pitch (F0) Extraction & Dynamics - Advanced Robust Version
    # ---------------------------------------------------------
    try:
        pitch = parselmouth.praat.call(sound, "To Pitch (cc)", 0.0, 50.0, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14,
                                       800.0)
        pitch_values = pitch.selected_array['frequency']

        if not hasattr(pitch_values, "__iter__") or isinstance(pitch_values, (float, int)):
            pitch_values = np.array([pitch_values])

        voiced_pitch = pitch_values[pitch_values > 0]
    except Exception:
        voiced_pitch = np.array([])

    if len(voiced_pitch) > 1:
        # FIX: Force extraction to standard float using float()
        features['mean_f0'] = float(np.mean(voiced_pitch))
        features['std_f0'] = float(np.std(voiced_pitch))
        features['pitch_velocity_std'] = float(np.std(np.diff(voiced_pitch / features['mean_f0'])))
    else:
        features['mean_f0'] = 0.0
        features['std_f0'] = 0.0
        features['pitch_velocity_std'] = 0.0

    # ---------------------------------------------------------
    # 2. Glottal Micro-Instability: Jitter & Shimmer
    # ---------------------------------------------------------
    try:
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        features['jitter'] = float(jitter) if not np.isnan(jitter) and jitter is not None else 0.0
        features['shimmer'] = float(shimmer) if not np.isnan(shimmer) and shimmer is not None else 0.0
    except Exception:
        features['jitter'] = 0.0
        features['shimmer'] = 0.0

    # ---------------------------------------------------------
    # 3. Vocal Tract Resonance: Formants (F1, F2, F3) - Robust Adaptive Version
    # ---------------------------------------------------------
    f1_list, f2_list, f3_list = [], [], []

    try:
        formant = sound.to_formant_burg(time_step=None, max_number_of_formants=5.0, maximum_formant=5500.0,
                                        window_length=0.025, pre_emphasis_from=50.0)
        num_frames = formant.get_number_of_frames()
        for frame in range(1, num_frames + 1):
            t = formant.get_time_from_frame_number(frame)
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if f1 is not None and not np.isnan(f1): f1_list.append(f1)
            if f2 is not None and not np.isnan(f2): f2_list.append(f2)
            if f3 is not None and not np.isnan(f3): f3_list.append(f3)

    except Exception:
        try:
            formant = sound.to_formant_burg(time_step=None, max_number_of_formants=5.0, maximum_formant=5000.0,
                                            window_length=0.035, pre_emphasis_from=50.0)
            num_frames = formant.get_number_of_frames()
            for frame in range(1, num_frames + 1):
                t = formant.get_time_from_frame_number(frame)
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                if f1 is not None and not np.isnan(f1): f1_list.append(f1)
                if f2 is not None and not np.isnan(f2): f2_list.append(f2)
                if f3 is not None and not np.isnan(f3): f3_list.append(f3)
        except Exception:
            pass

    # FIX: Ensure fallback values are clean float numbers, not mixed numpy objects
    features['mean_F1'] = float(np.mean(f1_list)) if f1_list else 0.0
    features['mean_F2'] = float(np.mean(f2_list)) if f2_list else 0.0
    features['mean_F3'] = float(np.mean(f3_list)) if f3_list else 0.0

    features['F2_F1_ratio'] = float(features['mean_F2'] / features['mean_F1']) if features['mean_F1'] > 0 else 0.0

    # Compile the 9 engineered biomarkers into a clean 1D vector
    feature_vector = [
        features['mean_f0'], features['std_f0'], features['pitch_velocity_std'],
        features['jitter'], features['shimmer'],
        features['mean_F1'], features['mean_F2'], features['mean_F3'],
        features['F2_F1_ratio']
    ]

    return np.array(feature_vector, dtype=float)