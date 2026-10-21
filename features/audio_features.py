import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import scipy.signal

"""
Some of the features are from the articles I was mentioning in the bibliography
For a summary of recurring parameters and coefficients, see the guide at the end of this page.
"""


def extract_linear_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    This function is optional to be used for CNN it extracts a Mel-Spectrogram on a linear amplitude scale
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    return mel_spec


def extract_mfcc_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts 13 MFCCs plus speed (Delta) and acceleration (Delta-Delta) features.
    Returns a 39-dimensional vector representing vocal tract shape and phonetic movement.

     The derivatives capture the dynamic speed and acceleration of frequency changes over time. This reveals the
    rhythm and phoneme transition patterns unique to each accent.
    """
    # Extract standard Mel-Frequency Cepstral Coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)

    # Check the total number of frames available along the time axis
    num_frames = mfcc.shape[1]

    # Safe boundary handling to prevent librosa delta estimation parameter errors
    if num_frames >= 9:
        # Standard configuration ,using default window width of 9 for optimal resolution
        delta_mfcc = librosa.feature.delta(mfcc, width=9)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=9)

    elif num_frames >= 3:
        # For short audio samples, we adaptively scale down
        # the window width (ensuring it remains an odd integer >= 3 ,(librosa.feature.delta demands odd number)
        smallest_width = 3
        if smallest_width % 2 == 0:
            smallest_width -= 1

        delta_mfcc = librosa.feature.delta(mfcc, width=smallest_width)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=smallest_width)

    else:
        #  edge-case: If audio is too short to estimate any derivatives,
        # fallback to zero-padding matrices to maintain downstream shape consistency
        delta_mfcc = np.zeros_like(mfcc)
        delta2_mfcc = np.zeros_like(mfcc)

    # Calculate global averages across the time axis to create a 39-dimensional vector
    mfcc_dynamic = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1)
    ])

    return mfcc, delta_mfcc, delta2_mfcc, mfcc_dynamic


def extract_pitch_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """
    Estimates the fundamental frequency (F0) using the YIN algorithm,
    and computes its global mean and standard deviation.

    F0 represents the melody and intonation of speech. Analyzing its standard deviation
    is essential for distinguishing native speakers from non-native speakers whose mother
    tongue is tonal (for example chinese), where pitch changes the meaning of words,
    which affects how they speak English.
    """
    f0 = librosa.yin(y=y, sr=sr, fmin=65, fmax=400, frame_length=2048, hop_length=512)

    # Filter out NaN values which represent unvoiced frames or silence,keeps only voiced frames
    voiced_f0 = f0[~np.isnan(f0)]

    # Handle edge cases where no voiced frames are detected (std = standard deviation)
    if len(voiced_f0) > 0:
        pitch_mean = float(np.mean(voiced_f0))
        pitch_std = float(np.std(voiced_f0))
    else:
        pitch_mean, pitch_std = 0.0, 0.0

    return f0, pitch_mean, pitch_std


def extract_spectral_centroid(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """
    Computes the Spectral Centroid for each frame and calculates its overall mean.

    This feature indicates the "center of mass" of the spectrum, reflecting the perceived
    brightness of the sound. It helps the model understand whether the acoustic energy of an
    accent is concentrated in higher or lower frequency ranges during articulation.
    """
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
    mean_centroid = float(np.mean(spec_centroid))
    return spec_centroid, mean_centroid


def extract_spectral_rolloff(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """
    Computes the Spectral Roll-off (85% energy cutoff frequency) and it's overall mean.

    This feature measures spectral asymmetry and helps distinguish between voiced sounds
    (low frequencies) and unvoiced sounds (high frequencies). Non-native speakers often
    modify the duration or structure of fricatives, which is captured by changes in roll-off.
    """
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=512, roll_percent=0.85)
    mean_rolloff = float(np.mean(spec_rolloff))
    return spec_rolloff, mean_rolloff


def extract_zero_crossing_rate(y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes the Zero Crossing Rate (ZCR) for each frame and calculates its overall mean.

    ZCR measures how frequently the audio signal changes its sign, acting as a direct proxy
    for noisiness. It is valuable for detecting unvoiced fricative consonants (like S or Th).
    Accents that struggle with pronouncing specific English fricatives show distinct ZCR patterns.
    The mean ZCR provides a single score showing how much friction noise a speaker
    uses overall across the entire recording
    """
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
    mean_zcr = float(np.mean(zcr))
    return zcr, mean_zcr


def extract_rms_features(y: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """
    Extracts Root Mean Square (RMS) energy per frame and evaluates its global
    mean, standard deviation, maximum, and minimum values.

    RMS represents volume intensity over time. Analyzing the feature standard deviation and variance
    reveals syllable stress patterns. This helps detect if a speaker keeps natural English word emphasis or
    uses their native language rhythm (common in languages like French or Spanish).
    """
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

    volume_variance = float(np.std(rms))
    volume_max = float(np.max(rms))
    volume_min = float(np.min(rms))
    average_volume = float(np.mean(rms))

    return rms, average_volume, volume_variance, volume_max, volume_min


def extract_duration_and_phonation(y: np.ndarray, sr: int, f0: np.ndarray, hop_length: int = 512) -> tuple[float, float, float]:
    """
    Computes total audio duration, active phonation time, and the overall phonation time ratio.
    Phonation time measures total actual speech duration to detect pauses, hesitations, and overall speech fluency.
    """
    total_duration = float(librosa.get_duration(y=y, sr=sr))

    # Isolate voiced frames where F0 is valid (not NaN and greater than 0)
    voiced_frames = f0[~np.isnan(f0) & (f0 > 0)]

    # Dynamic calculation based on the actual hop_length passed into the function
    phonation_time = float(len(voiced_frames) * (hop_length / sr))

    # Ensure phonation_time never visually exceeds total_duration due to framing round-offs
    if phonation_time > total_duration:
        phonation_time = total_duration

    # Calculate the ratio with a safety ceiling of 1.0
    phonation_ratio = float(phonation_time / total_duration) if total_duration > 0 else 0.0

    return total_duration, phonation_time, phonation_ratio

"""
end of regular features 
"""

"""
More advanced features 
"""

def extract_advanced_syllable_rate(y: np.ndarray, sr: int, phonation_time: float) -> float:
    """
    Estimates the speaking rate by detecting energetic peaks (center of a syllable)
    in the RMS envelope and normalizing by the active speech phonation time.

    This linguistic feature is vital for distinguishing between syllable-timed
    languages (e.g., Spanish, French) and stress-timed languages (e.g., native English).
    Non-native speakers typically carry over their mother tongue's rhythmic pace.

    In this function we used volume peaks to count spoken syllables while filtering
    out background noise and fast micro-fluctuations.
    We then divided the syllable count by net speech time (excluding pauses)
    to accurately measure true speaking speed.
    """
    # Isolate the 1D RMS energy envelope over time - the highes and lows of energy
    rms_envelope = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]

    # Calculate a dynamic energy threshold based on the average signal intensity
    mean_energy = np.mean(rms_envelope)

    # Detect local peaks representing the center vowel sound of each syllable
    # 'distance=5' ensures a realistic minimum duration between individual syllables (~160ms)
    peaks, _ = scipy.signal.find_peaks(rms_envelope, height=mean_energy * 0.5, distance=5)

    total_syllables = len(peaks)

    # Calculate syllable rate per second of active speech (prevent division by zero)
    syllable_rate = float(total_syllables / phonation_time) if phonation_time > 0 else 0.0

    return syllable_rate


def extract_vowel_consonant_ratio(f0: np.ndarray, zcr: np.ndarray, rms: np.ndarray) -> float:
    """
    Calculates the time ratio between vowel sounds and consonant sounds.
    Helps detect if a speaker stretches vowels or mispronounces consonants

    Non-native speakers often stretch vowels durations or struggle to properly articulate
    English consonants, altering the natural vowel-to-consonant phonetic balance.

    This function converts audio feature matrices into flat arrays and sets baseline thresholds for sound volume and friction noise.
    It counts frames with active pitch as vowels and frames with high friction noise as consonants.
    Finally, it divides the vowel frame count by the consonant frame count to measure the speaker's phonetic balance.
    """
    # Flatten multidimensional arrays to 1D for unified frame-by-frame filtering
    f0_flat = f0.flatten()
    zcr_flat = zcr.flatten()
    rms_flat = rms.flatten()

    # Set baseline thresholds for sound energy and friction noise
    energy_threshold = np.mean(rms_flat) * 0.2
    zcr_threshold = np.mean(zcr_flat)

    # Count vowels have active vocal cord vibration (F0 > 0) and sufficient volume
    vowel_frames = np.sum((~np.isnan(f0_flat)) & (f0_flat > 0) & (rms_flat > energy_threshold))

    # Count consonants lack vocal cord vibration but generate turbulent friction noise (High ZCR)
    consonant_frames = np.sum(((np.isnan(f0_flat)) | (f0_flat == 0)) & (zcr_flat > zcr_threshold))

    # Calculate the ratio (safeguard against division by zero)
    vowel_consonant_ratio = float(vowel_frames / consonant_frames) if consonant_frames > 0 else 0.0

    return vowel_consonant_ratio


def extract_formant_frequencies(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Estimates the average first and second formant frequencies (F1 and F2)
    using Linear Predictive Coding (LPC) coefficients over the speech signal.

    Acoustically, F1 correlates with tongue height (open/closed jaw), and F2
    correlates with tongue advancement (front/back position) inside the vocal tract trapezoid.
    Foreign accents typically display different formant patterns rooted in their native phonology.

    This function uses LPC polynomial roots to estimate resonant frequencies of the vocal tract.
    It converts these roots to Hertz, filters out noise,
    and extracts F1 (tongue height) and F2 (tongue position) to analyze vowel pronunciation.
    """
    # High-pass pre-emphasis filter to boost higher frequencies and stabilize LPC roots
    y_preemphasized = librosa.effects.preemphasis(y)

    # Determine the ideal LPC order based on the sampling rate (16KHZ)
    # 2 coefficients for each formant pair, plus 2-4 additional coefficients for the glottal source
    lpc_order = int(2 + sr / 1000)

    # Extract global LPC coefficients across the pre-emphasized signal
    lpc_coeffs = librosa.lpc(y_preemphasized, order=lpc_order)

    # Calculate the roots of the LPC polynomial equation to find peak frequencies
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]  # Keep only the positive imaginary half of the complex plane

    # Convert the raw polynomial roots back to physical frequency values in Hertz
    frequencies = np.arctan2(np.imag(roots), np.real(roots)) * (sr / (2 * np.pi))
    frequencies = np.sort(frequencies)  # Sort from lowest to highest frequency resonances

    # Isolate valid speech formants by filtering out the DC offset near 0Hz
    valid_formants = [f for f in frequencies if f > 90]

    # Safely extract F1 and F2, defaulting to standard human baseline averages if estimation fails
    f1 = float(valid_formants[0]) if len(valid_formants) > 0 else 500.0
    f2 = float(valid_formants[1]) if len(valid_formants) > 1 else 1500.0

    return f1, f2

"""
===============================================================================
AUDIO FEATURE EXTRACTION - RECURRING PARAMETERS & COEFFICIENTS GUIDE
===============================================================================

1. SIGNAL PROCESSING & FRAMING
-------------------------------------------------------------------------------
- sr (Sampling Rate):
  Number of audio samples per second (typically 16,000 Hz).

- n_fft = 2048:
  FFT Window Size (~128 ms at 16kHz). Converts time domain into frequency spectrum.

- hop_length = 512:
  Step size between consecutive frames (~32 ms). Creates a 75% frame overlap.

- frame_length = 2048 / 1024:
  Audio chunk size used for calculating time-domain features (RMS, ZCR, Pitch).

2. SPECTRAL & FEATURE BOUNDARIES
-------------------------------------------------------------------------------
- n_mels = 128:
  Number of Mel-frequency bands, mapping frequencies to human pitch perception.

- n_mfcc = 13:
  Number of MFCC coefficients extracted to represent vocal tract shape.

- width = 9 / 3:
  Time-window size (in frames) used to compute Delta speed and acceleration.

- fmin = 65 & fmax = 400:
  Search boundary in Hz for fundamental pitch (F0), covering human vocal ranges.

- roll_percent = 0.85:
  Cutoff percentage defining the frequency below which 85% of energy lies.

3. DYNAMIC DETECTION THRESHOLDS
-------------------------------------------------------------------------------
- height = mean_energy * 0.5:
  Energy threshold requiring peaks to reach at least 50% of mean audio volume.

- distance = 5:
  Minimum frame separation (~160 ms) between peaks to enforce physical syllable timing.

- energy_threshold = mean_rms * 0.2:
  Volume threshold (20% of mean RMS) used to filter out silent/background frames.

- f > 90:
  Frequency threshold filtering out non-speech DC offset frequencies below 90 Hz.

===============================================================================
"""