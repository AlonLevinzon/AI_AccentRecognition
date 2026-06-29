import os
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf


def normalize_audio_pipeline(input_path, output_path, target_sr=16000):
    """
    Runs a full acoustic normalization pipeline on an audio file
    to match its technical properties with other data sources.
    """
    try:
        # Step 1: Load audio, force to mono, and resample to the target sample rate
        # Using target_sr automatically converts the file during the loading phase
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)

        if len(y) == 0:
            print(f"Error: Empty or corrupted file at {input_path}")
            return False

        # Step 2: Remove stationary background noise (Noise Reduction)
        # The algorithm estimates the noise profile and attenuates it across the signal
        y_clean = nr.reduce_noise(y=y, sr=sr)

        # Step 3: Strip absolute silence from the beginning and end (Trimming)
        # Silence in different sources contains microphone 'color', so we remove it.
        # top_db=20 means any sound 20dB below the peak amplitude is considered silence.
        y_trimmed, _ = librosa.effects.trim(y_clean, top_db=20)

        # Step 4: Peak Amplitude Normalization
        # Scales the peak amplitude to 1.0 (or 0.95 to prevent digital clipping).
        # This eliminates volume differences caused by microphone distance or speech intensity.
        max_peak = np.max(np.abs(y_trimmed))
        if max_peak > 0:
            y_normalized = y_trimmed / max_peak
            # Multiply by 0.95 to leave a safe headroom margin
            y_normalized = y_normalized * 0.95
        else:
            y_normalized = y_trimmed

        # Step 5: Save the normalized audio into a standardized, uncompressed WAV format
        # Saving as 16-bit PCM WAV ensures no further compression artifacts are introduced
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_normalized, sr, subtype="PCM_16")
        return True

    except Exception as e:
        print(f"Exception encountered while processing {input_path}: {str(e)}")
        return False


# Usage Example:
if __name__ == "__main__":
    # Example input from an external unverified source (e.g., YouTube download)
    input_file = "../noisy_talking.wav"
    output_file = "../normalized_dataset/examples/noisy_talking.wav"

    # Execute pipeline (Target SR: 16kHz - standard for speech processing and ML)
    success = normalize_audio_pipeline(input_file, output_file, target_sr=16000)

    if success:
        print("Audio normalization pipeline completed successfully!")