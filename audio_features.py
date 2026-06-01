import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import scipy.signal
"""
this is the features from the 3 articles I was mentioning in the bibliography
"""


def extract_linear_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts a Mel-Spectrogram on a linear amplitude scale (without log/dB compression).

    This feature is crucial for deep learning models (such as CNNs) because preserving the
    linear amplitude scale avoids compressing phonetic data, allowing the neural network
    to directly learn fine-grained energy and pronunciation variance specific to foreign accents.
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    return mel_spec


def extract_mfcc_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts 13 MFCC coefficients, 1st derivative (Delta), and 2nd derivative (Delta-Delta),
    along with a concatenated time-averaged 39-dimensional feature vector.

    While standard MFCCs provide a static snapshot of the vocal tract filter, the derivatives
    capture the dynamic speed and acceleration of frequency changes over time. This reveals the
    rhythm and phoneme transition patterns unique to each accent.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Calculate global averages across the time axis to create a 39-dimensional vector
    mfcc_dynamic = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1)
    ])

    return mfcc, delta_mfcc, delta2_mfcc, mfcc_dynamic


def extract_pitch_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """
    Estimates the fundamental frequency (F0) using the YIN autocorrelation algorithm,
    and computes its global mean and standard deviation.

    F0 represents the melody and intonation of speech. Analyzing its standard deviation
    is essential for distinguishing native speakers from non-native speakers whose mother
    tongue is tonal (e.g., Chinese), where pitch fluctuations carry semantic meaning and
    spill over into English speech.
    """
    f0 = librosa.yin(y=y, sr=sr, fmin=65, fmax=400, frame_length=2048, hop_length=512)

    # Filter out NaN values which represent unvoiced frames or silence
    voiced_f0 = f0[~np.isnan(f0)]

    # Handle edge cases where no voiced frames are detected
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
    Computes the Spectral Roll-off (85% energy cutoff frequency) and its overall mean.

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
    """
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
    mean_zcr = float(np.mean(zcr))
    return zcr, mean_zcr


def extract_rms_features(y: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """
    Extracts Root Mean Square (RMS) energy per frame and evaluates its global
    mean, standard deviation, maximum, and minimum values.

    RMS represents volume intensity over time. Analyzing its standard deviation and variance
    reveals syllable stress patterns. This helps identify whether a speaker maintains natural
    English word-stress or applies the syllable-timed, uniform rhythm of their native tongue
    (e.g., French or Spanish).
    """
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

    volume_variance = float(np.std(rms))
    volume_max = float(np.max(rms))
    volume_min = float(np.min(rms))
    average_volume = float(np.mean(rms))

    return rms, average_volume, volume_variance, volume_max, volume_min


def extract_duration_and_phonation(y: np.ndarray, sr: int, f0: np.ndarray) -> tuple[float, float, float]:
    """
    Computes total audio duration, active phonation time (speech frames where vocal cords vibrate),
    and the overall phonation time ratio.

    These temporal metrics reflect speech rate and fluency. Non-native speakers often experience
    hesitation and introduce pause intervals to plan articulation, leading to a significantly
    lower phonation ratio compared to fluent or native speakers.
    """
    total_duration = float(librosa.get_duration(y=y, sr=sr))

    # Isolate voiced frames where F0 is valid (not NaN and greater than 0)
    voiced_frames = f0[~np.isnan(f0) & (f0 > 0)]

    # Calculate absolute speech time based on the hop length (512 samples per step)
    phonation_time = float(len(voiced_frames) * (512 / sr))

    # Calculate the ratio (safeguard against division by zero if audio is empty)
    phonation_ratio = float(phonation_time / total_duration) if total_duration > 0 else 0.0

    return total_duration, phonation_time, phonation_ratio

"""
end of basic of features 
"""

"""
advanced features 
"""

def extract_advanced_syllable_rate(y: np.ndarray, sr: int, phonation_time: float) -> float:
    """
    Estimates the speaking rate by detecting energetic peaks (syllable nuclei)
    in the RMS envelope and normalizing by the active speech phonation time.

    This linguistic feature is vital for distinguishing between syllable-timed
    languages (e.g., Spanish, French) and stress-timed languages (e.g., native English).
    Non-native speakers typically carry over their mother tongue's rhythmic pace.
    """
    # Isolate the 1D RMS energy envelope over time
    rms_envelope = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]

    # Calculate a dynamic energy threshold based on the average signal intensity
    mean_energy = np.mean(rms_envelope)

    # Detect local peaks representing the vowel nuclei of articulated syllables
    # 'distance=5' ensures a realistic minimum duration between individual syllables (~160ms)
    peaks, _ = scipy.signal.find_peaks(rms_envelope, height=mean_energy * 0.5, distance=5)

    total_syllables = len(peaks)

    # Calculate syllable rate per second of active speech (prevent division by zero)
    syllable_rate = float(total_syllables / phonation_time) if phonation_time > 0 else 0.0

    return syllable_rate


def extract_vowel_consonant_ratio(f0: np.ndarray, zcr: np.ndarray, rms: np.ndarray) -> float:
    """
    Computes the temporal ratio between voiced segments (vowels) and unvoiced/noisy
    fricative segments (consonants) by cross-referencing F0, ZCR, and RMS features.

    Non-native speakers often elongate vowel durations or struggle to properly articulate
    English consonants, altering the natural vowel-to-consonant phonetic balance.
    """
    # Flatten multi-dimensional arrays to 1D for unified frame-by-frame filtering
    f0_flat = f0.flatten()
    zcr_flat = zcr.flatten()
    rms_flat = rms.flatten()

    # Define acoustic baselines for voice presence and high frequency friction noise
    energy_threshold = np.mean(rms_flat) * 0.2
    zcr_threshold = np.mean(zcr_flat)

    # Rule 1: Vowels have active vocal cord vibration (F0 > 0) and sufficient volume
    vowel_frames = np.sum((~np.isnan(f0_flat)) & (f0_flat > 0) & (rms_flat > energy_threshold))

    # Rule 2: Consonants lack vocal cord vibration but generate turbulent friction noise (High ZCR)
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
    Foreign accents typically display rigid or shifted formant patterns rooted in their native phonology.
    """
    # High-pass pre-emphasis filter to boost higher frequencies and stabilize LPC roots
    y_preemphasized = librosa.effects.preemphasis(y)

    # Determine the ideal LPC order based on the sampling rate (standard formulation)
    # 2 coefficients for each formant pair, plus 2-4 additional coefficients for the glottal source
    lpc_order = int(2 + sr / 1000)

    # Extract global LPC coefficients across the pre-emphasized signal
    lpc_coeffs = librosa.lpc(y_preemphasized, order=lpc_order)

    # Calculate the roots of the LPC polynomial equation to find resonant peak frequencies
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
features from website medium
"""
def audio_wave_display():
    audio_path = "accents_samples/accent_archive/arabic/arabic45.mp3"

    try:
        #file load
        y, sr = librosa.load(audio_path, sr=None)

        #cleaning the noise
        cleaned_audio = nr.reduce_noise(y=y, sr=sr)

        # visualisation
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(cleaned_audio, sr=sr)
        plt.title("Cleaned Audio Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Not fount the path")

def MFCC_features():
    audio_path = "accents_samples/accent_archive/arabic/arabic45.mp3"
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)


        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Frequency (Hz)')
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("Not fount the path")

def pitch_Tempo_analysis():
    audio_path = "accents_samples/accent_archive/arabic/arabic45.mp3"

    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        pitch_values.append(pitch if pitch > 0 else np.nan)


    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]


    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(pitch_values, color='b')
    plt.title('Pitch Analysis')
    plt.xlabel('Frames')
    plt.ylabel('Pitch (Hz)')


    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.vlines(librosa.frames_to_time(beats, sr=sr), -1, 1, color='r', linestyle='--', label='Beats')
    plt.title(f'Tempo Analysis (Tempo: {tempo:.2f} BPM)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')


    plt.tight_layout()
    plt.show()


    print(f"Estimated Tempo: {tempo:.2f} BPM")

# def main():
#     #audio_graph()
#     MFCC_features()
#     #pitch_Tempo_analysis()
#
#
# if __name__ == "__main__":
#     main()