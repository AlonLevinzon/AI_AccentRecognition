import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path


def process_raw_audio_pipeline(file_path, target_sr=16000):
    """
    Safe Audio Normalization Pipeline:
    1. Gentle trimming (top_db=40) to preserve prosody, pauses, and soft consonants.
    2. RMS Normalization (target_rms=0.05) to ensure consistent power without distortion.
    3. Excludes aggressive noise reduction to protect spectral formants and MFCCs.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)

        if len(y) / sr < 0.1:
            return None

        # 1. Gentle Trimming (top_db=40 preserves soft fricatives and speech rhythm)
        y_trimmed, _ = librosa.effects.trim(y, top_db=40)
        if len(y_trimmed) / sr < 0.1:
            return None

        # 2. RMS Normalization (Safer for Speech/Accent Features than Peak Normalization)
        rms = np.sqrt(np.mean(y_trimmed**2))
        if rms > 0:
            target_rms = 0.05
            y_normalized = y_trimmed * (target_rms / rms)
            # Clip to prevent digital distortion
            y_normalized = np.clip(y_normalized, -0.99, 0.99)
        else:
            return None

        return y_normalized

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def verify_normalization_results(target_dirs, max_allowed_failures=5):
    """
    Verifies that all processed audio files satisfy RMS criteria and length constraints.
    """
    print("\n🔍 Running automatic post-normalization verification check (RMS criteria)...")
    failed_files = []
    total_checked = 0

    for check_dir in target_dirs:
        check_path = Path(check_dir)
        if not check_path.exists():
            continue

        wav_files = [f for f in check_path.rglob("*") if f.name.lower().endswith(".wav")]
        print(f"📁 Checking {len(wav_files)} files in: {check_path.name}...")

        for file_path in wav_files:
            total_checked += 1
            try:
                y, sr = librosa.load(file_path, sr=16000, mono=True)
                if len(y) == 0:
                    failed_files.append((str(file_path), "Empty audio file"))
                    continue

                duration = len(y) / sr
                if duration < 0.1:
                    failed_files.append((str(file_path), f"File too short: {duration:.3f}s"))
                    continue

                rms_val = np.sqrt(np.mean(y**2))
                if rms_val < 0.02 or rms_val > 0.15:
                    failed_files.append((str(file_path), f"Invalid RMS Energy: {rms_val:.4f}"))
                    continue

            except Exception as e:
                failed_files.append((str(file_path), f"Load error: {str(e)}"))

    failed_count = len(failed_files)
    print(f"📊 Total files verified: {total_checked} | Total failures: {failed_count}")

    if failed_count > max_allowed_failures:
        raise ValueError(f"❌ [VALIDATION FAILED]: Found {failed_count} unnormalized files!")

    print(f"🎉 [VALIDATION PASSED]: Verified successfully ({failed_count} failures <= {max_allowed_failures}).")


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent
    verify_normalization_results([
        PROJECT_DIR / "sentences_samples_normalized",
        PROJECT_DIR / "audio_by_accent_raw_normalized"
    ])