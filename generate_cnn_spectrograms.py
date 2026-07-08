import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_and_save_spectrograms(metadata_path, audio_dir, output_dir, target_seconds=2.0, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(metadata_path):
        print(f"❌ CRITICAL ERROR: Metadata not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    target_length = int(target_seconds * sr)
    
    # Track success/failure for the console report
    success_count = 0
    missing_count = 0
    
    print(f"\n📁 Target Audio Directory: {audio_dir}")
    print(f"📁 Target Output Directory: {output_dir}")
    print(f"📄 Target Metadata File: {metadata_path}")
    print("-" * 50)

    for filename in tqdm(df['filename'], desc=f"Processing {os.path.basename(audio_dir)}"):
        audio_path = os.path.join(audio_dir, filename)
        
        if not os.path.exists(audio_path):
            missing_count += 1
            # If a file is missing, print a warning but don't crash
            # print(f"Warning: Audio file not found -> {audio_path}")
            continue
            
        try:
            # 1. Load Audio
            y, _ = librosa.load(audio_path, sr=sr)
            
            # 2. Truncate or Pad (To exactly 2.0 seconds)
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
                
            # 3. Linear Scale Mel-Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, power=1.0)
            
            # 4. Z-Score Normalization
            mean = np.mean(mel_spec)
            std = np.std(mel_spec)
            if std > 0:
                mel_spec = (mel_spec - mean) / std
                
            # 5. Save exactly as (64, 63) Numpy Array
            save_name = os.path.splitext(filename)[0] + '.npy'
            np.save(os.path.join(output_dir, save_name), mel_spec)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            missing_count += 1

    print("-" * 50)
    print(f"✅ {os.path.basename(audio_dir).upper()} Summary: {success_count} generated successfully | {missing_count} missing/failed.")


def main():
    base_dir = "official_audio_mixed"
    out_base = "official_cnn_spectrograms"
    
    splits = ['train', 'validation', 'test']
    for split in splits:
        print(f"\n{'='*50}\n--- INITIALIZING SPECTROGRAM PIPELINE: {split.upper()} ---\n{'='*50}")
        meta_path = f"{base_dir}/{split}_metadata.csv"
        audio_dir = f"{base_dir}/{split}"
        out_dir = f"{out_base}/{split}"
        
        process_and_save_spectrograms(meta_path, audio_dir, out_dir)
        
    print("\n🚀 ALL PIPELINES COMPLETED.")

if __name__ == '__main__':
    main()