import numpy as np
import parselmouth

"""
For a summary of recurring parameters and coefficients, see the guide at the end of this page.
"""

def extract_advanced_acoustic_features(audio_path):
    """
    Extracts physical and dynamic voice features based on Premananth et al. (2025).
    Guarantees all outputs are standard Python floats to prevent numpy dimension mismatch errors.

    The first block uses Praat to extract pitch (F0) and measures frame-by-frame pitch differences to capture pitch velocity.
    It then computes mean pitch, pitch stability, and pitch velocity variation (std) to analyze sharp intonation jumps.
    The second block calculate Jitter and Shimmer
    The third block extracts F1, F2, and F3 formants using Praat method with an adaptive fallback for low-pitch voices.
    It then computes mean formant frequencies and the F2/F1 ratio,
     assembling all 9 acoustic biomarkers into a final 1D feature vector.
    """
    features = {}

    # Load audio file into Praat sound object for precise acoustic analysis
    sound = parselmouth.Sound(str(audio_path))

    # ---------------------------------------------------------
    # 1. Pitch (F0) Extraction & Dynamics
    # ---------------------------------------------------------
    # Extract pitch contour using Praat's cross-correlation method (50Hz to 800Hz)
    try:
        pitch = parselmouth.praat.call(sound, "To Pitch (cc)", 0.0, 50.0, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14,
                                       800.0)
        pitch_values = pitch.selected_array['frequency']
        # Convert to numpy array and keep only active pitch values (greater than 0)
        if not hasattr(pitch_values, "__iter__") or isinstance(pitch_values, (float, int)):
            pitch_values = np.array([pitch_values])

        voiced_pitch = pitch_values[pitch_values > 0]
    except Exception:
        voiced_pitch = np.array([])
    # Calculate mean pitch, pitch variation (std), and pitch velocity (speed of pitch changes)
    if len(voiced_pitch) > 1:
        # Force extraction to standard float using float()
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
        # Locate exact vocal cord pulse points in time
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        # Measure pitch instability between consecutive vocal pulses (Jitter)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        # Measure volume instability between consecutive vocal pulses (Shimmer)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        features['jitter'] = float(jitter) if not np.isnan(jitter) and jitter is not None else 0.0
        features['shimmer'] = float(shimmer) if not np.isnan(shimmer) and shimmer is not None else 0.0
    except Exception:
        features['jitter'] = 0.0
        features['shimmer'] = 0.0

    # ---------------------------------------------------------
    # 3. Vocal Tract Resonance: Formants (F1, F2, F3)
    # ---------------------------------------------------------
    f1_list, f2_list, f3_list = [], [], []

    try:
        # Extract formants using Praat Burg algorithm with standard 5500 Hz cutoff
        formant = sound.to_formant_burg(time_step=None, max_number_of_formants=5.0, maximum_formant=5500.0,
                                        window_length=0.025, pre_emphasis_from=50.0)
        num_frames = formant.get_number_of_frames()
        for frame in range(1, num_frames + 1):
            t = formant.get_time_from_frame_number(frame) # converts frame to time
            f1 = formant.get_value_at_time(1, t) # gets formant 1 at time t
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if f1 is not None and not np.isnan(f1): f1_list.append(f1)
            if f2 is not None and not np.isnan(f2): f2_list.append(f2)
            if f3 is not None and not np.isnan(f3): f3_list.append(f3)

    except Exception:
        try:
            # Fallback attempt with wider window (35ms) and lower cutoff (5000Hz) if main extraction fails
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

    # Compute mean formants and calculate F2/F1 ratio for vowel articulation shape
    features['mean_F1'] = float(np.mean(f1_list)) if f1_list else 0.0
    features['mean_F2'] = float(np.mean(f2_list)) if f2_list else 0.0
    features['mean_F3'] = float(np.mean(f3_list)) if f3_list else 0.0

    features['F2_F1_ratio'] = float(features['mean_F2'] / features['mean_F1']) if features['mean_F1'] > 0 else 0.0

    # Compile the 9 engineered features into a clean 1D vector
    feature_vector = [
        features['mean_f0'], features['std_f0'], features['pitch_velocity_std'],
        features['jitter'], features['shimmer'],
        features['mean_F1'], features['mean_F2'], features['mean_F3'],
        features['F2_F1_ratio']
    ]

    return np.array(feature_vector, dtype=float)


"""
===============================================================================
PARSELMOUTH / PRAAT INTERNAL FUNCTIONS & PARAMETERS GUIDE
===============================================================================

1. parselmouth.Sound(audio_path)
   - Converts the audio file into a Praat object for precise sound analysis.

2. parselmouth.praat.call(sound, "To Pitch (cc)", ...)
   - Extracts pitch (F0) across human vocal limits:
     * 50.0 (pitch_floor): Lowest pitch to search for in Hz.
     * 800.0 (pitch_ceiling): Highest pitch to search for in Hz.
     * 15 (max_candidates): Number of pitch guesses tracked per frame.
     * "yes" (very_accurate): Turns on high-accuracy mode.

3. parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
   - Finds the exact time points where the vocal cords close in each cycle.

4. parselmouth.praat.call(point_process, "Get jitter (local)", ...)
   - Measures pitch instability (Jitter) between consecutive vocal pulses:
     * 0.0, 0.0: Analyzes the entire audio from start to end.
     * 0.0001 (shortest_period): Shortest allowed pulse duration (0.1 ms).
     * 0.02 (longest_period): Longest allowed pulse duration (20 ms).
     * 1.3 (max_period_factor): Ignores unnatural pitch jumps (over 1.3x difference).

5. parselmouth.praat.call([sound, point_process], "Get shimmer (local)", ...)
   - Measures volume instability (Shimmer) between consecutive vocal pulses:
     * Same settings as jitter, plus 1.6 (max_amplitude_factor) to ignore 
       unnatural volume spikes (over 1.6x difference).

6. sound.to_formant_burg(...)
   - Finds vocal tract resonant frequencies (F1, F2, F3):
     * max_number_of_formants=5.0: Searches for up to 5 resonant peaks.
     * maximum_formant=5500.0 / 5000.0: Max frequency ceiling in Hz 
       (5500 Hz for average voices, 5000 Hz fallback for deeper voices).
     * window_length=0.025 / 0.035: Frame size in seconds 
       (25 ms standard, 35 ms fallback for smoother tracking).
     * pre_emphasis_from=50.0: Filters out low noise below 50 Hz.

7. formant.get_number_of_frames() & formant.get_time_from_frame_number(frame)
   - Counts total frames and converts frame index to exact time in seconds.

8. formant.get_value_at_time(formant_number, time)
   - Gets the exact formant frequency in Hz (1=F1, 2=F2, 3=F3) at a specific time.
===============================================================================
"""