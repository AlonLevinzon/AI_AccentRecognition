import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path


# 1. Single audio file processing function (RMS Normalization)
def process_raw_audio_pipeline(file_path, target_sr=16000):
    """
    Loads an audio file, trims silence, and applies RMS normalization.
    """
    try:
        # Load audio file with specified sampling rate and mono channel
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)

        # Skip files shorter than 0.1 seconds
        if len(y) / sr < 0.1:
            return None

        # Lenient silence trimming (top_db=40)
        y_trimmed, _ = librosa.effects.trim(y, top_db=40)
        if len(y_trimmed) / sr < 0.1:
            return None

        # RMS Normalization to 0.05 -so the audio isn't too loud or too quiet.
        # This keeps speech volume consistent and prevents sound distortion.
        rms = np.sqrt(np.mean(y_trimmed ** 2))
        if rms > 0:
            target_rms = 0.05
            y_normalized = y_trimmed * (target_rms / rms)
            # Clip peak values to prevent digital clipping
            y_normalized = np.clip(y_normalized, -0.99, 0.99)
        else:
            return None

        return y_normalized

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# 2. Directory scanner and batch normalization runner
def run_mass_normalization(base_input_dir, output_dir):
    """
    Scans the base directory for target folders and applies audio normalization.
    """
    base_input_path = Path(base_input_dir)
    output_path = Path(output_dir)

    sentences_samples_src = None
    audio_by_accent_raw_src = None

    # Walk directory tree to find target source folders
    for root, dirs, _ in os.walk(base_input_path):
        for d in dirs:
            if d.lower() == "sentences_samples":
                sentences_samples_src = Path(root) / d
            elif d.lower() == "audio_by_accent_raw":
                audio_by_accent_raw_src = Path(root) / d

    # Map source directories to normalized output paths
    mapped_targets = {}
    if sentences_samples_src:
        mapped_targets[sentences_samples_src] = output_path / "sentences_samples_normalized"
    if audio_by_accent_raw_src:
        mapped_targets[audio_by_accent_raw_src] = output_path / "audio_by_accent_raw_normalized"

    print(" Starting Audio Normalization...")

    # Iterate through each source directory, normalize, and save output
    for src_dir, target_dir in mapped_targets.items():
        print(f"\n Processing directory: {src_dir.name} -> {target_dir.name}")

        for root, dirs, files in os.walk(src_dir):
            rel_path = Path(root).relative_to(src_dir)
            current_target_dir = target_dir / rel_path
            os.makedirs(current_target_dir, exist_ok=True)

            for file in files:
                src_file_path = Path(root) / file
                target_file_path = current_target_dir / file

                # Process WAV audio files; copy non-WAV files as-is
                if file.lower().endswith(".wav"):
                    y_normalized = process_raw_audio_pipeline(str(src_file_path), target_sr=16000)

                    if y_normalized is not None:
                        sf.write(str(target_file_path), y_normalized, 16000)
                else:
                    shutil.copy2(src_file_path, target_file_path)


# 3. Post-processing verification function
def verify_normalization_results(target_dirs, max_allowed_failures=5):
    """
    Verifies WAV files in output directories to ensure valid audio quality and RMS parameters.
    """
    print("\n Running automatic post-normalization verification check...")
    failed_files = []
    total_checked = 0

    for check_dir in target_dirs:
        check_path = Path(check_dir)
        if not check_path.exists():
            continue

        wav_files = [f for f in check_path.rglob("*") if f.name.lower().endswith(".wav")]
        print(f" Checking {len(wav_files)} files in: {check_path.name}...")

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

                rms_val = np.sqrt(np.mean(y ** 2))
                if rms_val < 0.02 or rms_val > 0.15:
                    failed_files.append((str(file_path), f"Invalid RMS Energy: {rms_val:.4f}"))
                    continue

            except Exception as e:
                failed_files.append((str(file_path), f"Load error: {str(e)}"))

    failed_count = len(failed_files)
    print(f" Total files verified: {total_checked} | Total failures: {failed_count}")

    if failed_count > max_allowed_failures:
        raise ValueError(f" [VALIDATION FAILED]: Found {failed_count} unnormalized files!")

    print(f" [VALIDATION PASSED]: Verified successfully.")


if __name__ == "__main__":
    # Get the project root directory
    PROJECT_DIR = Path(__file__).resolve().parent.parent

    # 1. Run normalization process on target directories
    run_mass_normalization(base_input_dir=PROJECT_DIR, output_dir=PROJECT_DIR)

    # 2. Verify that normalized output directories are valid
    verify_normalization_results([
        PROJECT_DIR / "sentences_samples_normalized",
        PROJECT_DIR / "audio_by_accent_raw_normalized"
    ])