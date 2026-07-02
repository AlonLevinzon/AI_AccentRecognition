import numpy as np
import librosa
import noisereduce as nr

# Internal tracking counter for short audio files
_short_audio_discard_counter = 0


def process_raw_audio_pipeline(input_path, target_sr=16000, min_sec=0.1):
    """
    Processes raw audio by filtering out short samples (< 3s), reducing noise,
    and normalizing amplitude. Keeps long files (> 7s) as they are without padding.
    Prints a status update directly whenever a file is discarded.

    Parameters:
    - input_path: str, path to the audio file.
    - target_sr: int, sample rate for resampling (default 16kHz).
    - min_sec: float, minimum duration threshold in seconds (default 3.0s).

    Returns:
    - y_normalized: np.ndarray, the cleaned raw audio signal, or None if discarded.
    """
    global _short_audio_discard_counter

    # ----------------------------------------------------
    # 1. Load, Mono & Resample
    # ----------------------------------------------------
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    if len(y) == 0:
        return None

    # ----------------------------------------------------
    # 2. Advanced Noise Reduction & Trimming
    # ----------------------------------------------------
    y_clean = nr.reduce_noise(y=y, sr=sr)
    y_trimmed, _ = librosa.effects.trim(y_clean, top_db=25)

    # ----------------------------------------------------
    # 3. Duration Filtering (Under 3 seconds only)
    # ----------------------------------------------------
    duration_sec = len(y_trimmed) / target_sr

    if duration_sec < min_sec:
        _short_audio_discard_counter += 1
        print(f"[DISCARDED] File too short ({duration_sec:.2f}s): {input_path}")
        print(f"Total short files discarded so far: {_short_audio_discard_counter}")
        return None

    # ----------------------------------------------------
    # 4. Peak Amplitude Normalization
    # ----------------------------------------------------
    max_peak = np.max(np.abs(y_trimmed))
    if max_peak > 0:
        y_normalized = (y_trimmed / max_peak) * 0.95
    else:
        y_normalized = y_trimmed

    return y_normalized