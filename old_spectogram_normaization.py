"""
audio_normalization.py
----------------------
Audio pre-processing module for accent classification.
Handles resampling to 16kHz, mono conversion, fixed-length padding/trimming,
and linear-scale amplitude Mel-spectrogram extraction.
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def process_and_extract_mel_spectrogram(
    audio_path: str,
    target_sr: int = 16000,
    target_duration_sec: float = 4.0,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """Processes an audio file and returns a linear-scale amplitude Mel-spectrogram.

    Parameters:
    -----------
    audio_path : str
        Path to the audio file.
    target_sr : int
        Target sample rate (default: 16000 Hz).
    target_duration_sec : float
        Target window duration in seconds (e.g., 4.0 seconds).
    n_mels : int
        Number of Mel frequency bins (spectrogram height).
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length (stride) between FFT windows.

    Returns:
    --------
    torch.Tensor: Tensor of shape [1, n_mels, time_steps]
    """
    # 1. Load audio file
    waveform, sr = torchaudio.load(audio_path)

    # 2. Convert to mono channel
    # If stereo or multi-channel, average across all channels
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 3. Resample to target sample rate (16kHz)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # 4. Enforce fixed length (Padding / Trimming)
    target_length = int(target_sr * target_duration_sec)
    num_samples = waveform.shape[1]

    if num_samples < target_length:
        # Zero-padding if audio is shorter than target length
        pad_amount = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > target_length:
        # Trim if audio is longer than target length
        waveform = waveform[:, :target_length]

    # 5. Extract Linear-Scale Amplitude Mel-Spectrogram
    # power=1.0 ensures linear amplitude scale (preserves phonetic subtleties)
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0,  # Critical requirement: Linear Scale Amplitude
    )

    mel_spec = mel_spectrogram_transform(waveform)

    # Standard Z-score normalization for numerical stability
    mean = mel_spec.mean()
    std = mel_spec.std() + 1e-6
    mel_spec = (mel_spec - mean) / std

    return mel_spec  # Shape: [1, n_mels, time_steps]


if __name__ == "__main__":
    # Local test execution
    test_path = "example_audio.wav"
    try:
        spec = process_and_extract_mel_spectrogram(test_path)
        print(f"Spectrogram extracted successfully! Shape: {spec.shape}")
    except Exception as e:
        print(f"Notice: Provide a valid file path to test. ({e})")