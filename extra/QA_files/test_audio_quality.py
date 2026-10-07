import os
import glob
import random
import requests
import librosa
import numpy as np
import pandas as pd
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------------------
# 1. Automatic Download of the Official DNSMOS Model (ONNX format)
# -------------------------------------------------------------------------
DNSMOS_MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
MODEL_PATH = "sig_bak_ovr.onnx"


def download_model_if_needed():
    """Downloads the DNSMOS weights if they do not exist locally."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading official DNSMOS model weights (First time only)...")
        response = requests.get(DNSMOS_MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded and saved locally.")


# -------------------------------------------------------------------------
# 2. Pure Measurement Function (NO AUDIO ALTERATION OR WRITING)
# -------------------------------------------------------------------------
def evaluate_audio_quality_only(file_path: str, session, dataset_name: str, sub_accent: str = "N/A") -> dict:
    """
    CRITICAL NOTE: THIS FUNCTION IS STRICTLY READ-ONLY.
    IT DOES NOT ALTER, ENHANCE, FIX, OR WRITE ANY AUDIO DATA ON DISK.
    It pads/crops the array inside RAM memory to match the strict 144160 inputs requirement.
    """
    try:
        # Load audio natively at 16kHz as required by DNSMOS (mono=True)
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(y) == 0:
            return None

        REQUIRED_LEN = 144160
        actual_len = len(y)

        # Adjust dimensions safely in memory to satisfy the rigid ONNX model constraint
        if actual_len < REQUIRED_LEN:
            pad_len = REQUIRED_LEN - actual_len
            y_fixed = np.pad(y, (0, pad_len), mode='constant')
        elif actual_len > REQUIRED_LEN:
            start_idx = (actual_len - REQUIRED_LEN) // 2
            y_fixed = y[start_idx:start_idx + REQUIRED_LEN]
        else:
            y_fixed = y

        # Ensure array format matches float32 and 2D batch dimension: [Batch=1, 144160]
        audio_input = np.expand_dims(y_fixed.astype(np.float32), axis=0)

        inputs = {session.get_inputs()[0].name: audio_input}
        raw_outputs = session.run(None, inputs)

        # Dynamically adapt to the ONNX model output structure
        if len(raw_outputs) >= 3:
            sig_score = float(np.mean(raw_outputs[0]))
            bak_score = float(np.mean(raw_outputs[1]))
            ovr_score = float(np.mean(raw_outputs[2]))
        else:
            flattened_output = np.squeeze(raw_outputs[0])
            sig_score = float(flattened_output[0])
            bak_score = float(flattened_output[1])
            ovr_score = float(flattened_output[2])

        return {
            "dataset_source": dataset_name,
            "sub_accent": sub_accent,
            "filename": os.path.basename(file_path),
            "speech_quality_sig": round(sig_score, 2),
            "background_noise_bak": round(bak_score, 2),
            "overall_quality_ovrl": round(ovr_score, 2)
        }
    except Exception as e:
        print(f"❌ Error processing file {os.path.basename(file_path)}: {str(e)}")
        return {
            "dataset_source": dataset_name,
            "sub_accent": sub_accent,
            "filename": os.path.basename(file_path),
            "speech_quality_sig": np.nan,
            "background_noise_bak": np.nan,
            "overall_quality_ovrl": np.nan
        }


# -------------------------------------------------------------------------
# 3. Parallel Scanning Framework Core with Sampling Optimization
# -------------------------------------------------------------------------
def scan_and_benchmark_all(all_accents_data_path: str, accents_audio_path: str, max_workers: int = 8,
                           sample_limit: int = 100):
    """
    Scans paths using os.walk and applies a random sampling cap to optimize execution speed.
    Leverages GPU acceleration if hardware providers are available.
    """
    download_model_if_needed()

    # Enable hardware acceleration (CUDA/GPU) if available, fallback to CPU safely
    available_providers = ort.get_available_providers()
    print(f"🤖 Available Execution Providers: {available_providers}")

    providers = ['CUDAExecutionProvider',
                 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    print(f"🚀 Using provider: {providers[0]}")

    opts = ort.SessionOptions()
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)

    all_results = []
    tasks = []
    valid_extensions = ('.wav', '.WAV', '.mp3', '.MP3', '.m4a', '.M4A', '.flac', '.FLAC')

    # Fixed seed for sampling reproducibility
    random.seed(42)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # ─── PART 1: SCAN STANDARD DATASETS IN ALL_ACCENTS_DATA (FLAT STRUCTURE) ──────
        if os.path.exists(all_accents_data_path):
            source_folders = [f.path for f in os.scandir(all_accents_data_path) if f.is_dir()]

            for folder in source_folders:
                source_name = os.path.basename(folder)
                audio_files = []

                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(valid_extensions):
                            audio_files.append(os.path.join(root, file))

                total_found = len(audio_files)
                # Apply sample limit cap to prevent execution bottlenecks
                if total_found > sample_limit:
                    audio_files = random.sample(audio_files, sample_limit)
                    print(
                        f"🔍 Found {total_found} files in '{source_name}' -> Sampling {len(audio_files)} files for rapid baseline.")
                else:
                    print(f"🔍 Found {total_found} files in '{source_name}' -> Processing all.")

                for file_path in audio_files:
                    tasks.append(
                        executor.submit(evaluate_audio_quality_only, file_path, session, source_name, "N/A")
                    )
        else:
            print(f"⚠️ Path not found: {all_accents_data_path}")

        # ─── PART 2: SCAN ACCENT_ARCHIVE UNDER ACCENTS_AUDIO (ACCENT STRUCTURED) ──────
        accent_archive_base = os.path.join(accents_audio_path, "accent_archive")
        if os.path.exists(accent_archive_base):
            accent_folders = [f.path for f in os.scandir(accent_archive_base) if f.is_dir()]

            for accent_folder in accent_folders:
                sub_accent_name = os.path.basename(accent_folder)
                audio_files = []

                for root, _, files in os.walk(accent_folder):
                    for file in files:
                        if file.endswith(valid_extensions):
                            audio_files.append(os.path.join(root, file))

                total_found = len(audio_files)
                # Apply sample limit cap to the accent archive sub-folders as well
                if total_found > sample_limit:
                    audio_files = random.sample(audio_files, sample_limit)
                    print(
                        f"🔍 Found {total_found} files under accent_archive -> '{sub_accent_name}' -> Sampling {len(audio_files)} files.")
                else:
                    print(f"🔍 Found {total_found} files under accent_archive -> '{sub_accent_name}' -> Processing all.")

                for file_path in audio_files:
                    tasks.append(
                        executor.submit(evaluate_audio_quality_only, file_path, session, "accent_archive",
                                        sub_accent_name)
                    )
        else:
            print(f"⚠️ Accent Archive root path not found: {accent_archive_base}")

        # ─── COLLECT ALL CONCURRENT SCORES AS THEY COMPLETE ───────────────────────────
        for future in tasks:
            res = future.result()
            if res is not None:
                all_results.append(res)

    # Build metrics report dataframe
    df_results = pd.DataFrame(all_results)

    # Print out macro aggregated performance stats cleanly to standard output console
    print("\n=======================================================")
    print("📊 Summary of Raw Audio Quality Scores (Scale 1.0 - 5.0):")
    print("=======================================================")
    if not df_results.empty:
        summary = df_results.groupby(["dataset_source", "sub_accent"])[
            ["speech_quality_sig", "background_noise_bak", "overall_quality_ovrl"]].mean()
        print(summary)
    else:
        print("❌ No valid results collected.")

    return df_results


# -------------------------------------------------------------------------
# 4. Main Running Context Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    ALL_ACCENTS_DATA_DIR = "../../all_accents_data"
    ACCENTS_AUDIO_DIR = "../../accents_audio"

    print("🚀 Starting optimized, highly accelerated non-destructive benchmarking...")
    # Cap limit set to 100 files per group for sub-minute execution
    df_final_report = scan_and_benchmark_all(ALL_ACCENTS_DATA_DIR, ACCENTS_AUDIO_DIR, max_workers=8, sample_limit=100)

    df_final_report.to_csv("all_raw_datasets_quality_report.csv", index=False)
    print("\n💾 Complete read-only evaluation report written out to: all_raw_datasets_quality_report.csv")