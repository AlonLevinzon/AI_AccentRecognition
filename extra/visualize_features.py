import matplotlib.pyplot as plt
import numpy as np
import parselmouth  # זהו המנוע של Praat בפייתון
import librosa
import librosa.display

# --- 1. טעינת הקובץ והגדרת זמנים (שנה את הפרמטרים כאן לפי הצורך) ---
audio_path = "audio_by_accent_raw_normalized/chinese/000030012.WAV"
start_time = 0.0  # תחילת הקיטוע בשניות
end_time = 3.0    # סוף הקיטוע בשניות

# טעינת האודיו ל-Praat ול-Librosa
sound = parselmouth.Sound(audio_path).extract_part(from_time=start_time, to_time=end_time)
y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=(end_time - start_time))

# --- 2. חילוץ מדדים באמצעות מנוע Praat ---
pitch = sound.to_pitch()
pitch_values = pitch.selected_array['frequency']
pitch_values[pitch_values == 0] = np.nan  # העלמת ערכי אפס (שקט) בשביל הגרף
pitch_times = pitch.xs() + start_time

intensity = sound.to_intensity()
intensity_values = intensity.values[0]
intensity_times = intensity.xs() + start_time

formants = sound.to_formant_burg()
formant_times = formants.xs() + start_time
f1_values = [formants.get_value_at_time(1, t) for t in formants.xs()]
f2_values = [formants.get_value_at_time(2, t) for t in formants.xs()]

# --- 3. חילוץ מדדים ספקטרליים נוספים באמצעות Librosa ---
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
librosa_times = librosa.times_like(centroid, sr=sr) + start_time

# --- 4. יצירת מערך הגרפים לדוח הפרויקט ---
fig, axs = plt.subplots(6, 1, figsize=(12, 16), sharex=False)

# גרף 1: Waveform + Zero Crossing Rate

# Waveform
librosa.display.waveshow(
    y,
    sr=sr,
    ax=axs[0],
    axis='time',
    x_axis='time',
    alpha=0.5
)

axs[0].set_ylabel("Amplitude")

# חישוב ZCR
zcr = librosa.feature.zero_crossing_rate(y=y)[0]
zcr_times = librosa.times_like(zcr, sr=sr)

# ציר Y נוסף עבור ZCR
ax_zcr = axs[0].twinx()

ax_zcr.plot(
    zcr_times,
    zcr,
    linewidth=1.5,
    label="ZCR"
)

ax_zcr.set_ylabel("Zero Crossing Rate")
ax_zcr.set_ylim(0, 1)

axs[0].set_title(
    "1. Waveform and Zero Crossing Rate (ZCR)",
    fontsize=12,
    fontweight='bold'
)

ax_zcr.legend(loc='upper right')

# גרף 2: Pitch Contour (Praat F0)
axs[1].plot(pitch_times, pitch_values, color='blue', linewidth=2)
axs[1].set_title("2. Pitch Contour / Fundamental Frequency (F₀ - Praat)", fontsize=12, fontweight='bold')
axs[1].set_ylabel("Frequency (Hz)")
axs[1].grid(True)

# גרף 3: Intensity (Praat RMS/Loudness)
axs[2].plot(intensity_times, intensity_values, color='green', linewidth=2)
axs[2].set_title("3. Intensity / Loudness Envelope (RMS - Praat)", fontsize=12, fontweight='bold')
axs[2].set_ylabel("Amplitude (dB)")
axs[2].grid(True)

# גרף 4: Formants (Praat F1 & F2)
axs[3].plot(formant_times, f1_values, label='F1 (Tongue Height)', color='red', linestyle='--')
axs[3].plot(formant_times, f2_values, label='F2 (Tongue Advancement)', color='darkred', linewidth=2)
axs[3].set_title("4. Vocal Tract Formants (F₁ & F₂ - Praat)", fontsize=12, fontweight='bold')
axs[3].set_ylabel("Frequency (Hz)")
axs[3].legend(loc="upper right")
axs[3].grid(True)

# גרף 5: Spectral Centroid
axs[4].plot(librosa_times, centroid[0], color='orange', linewidth=2)
axs[4].set_title("5. Spectral Centroid (Voice Brightness / Center of Mass)", fontsize=12, fontweight='bold')
axs[4].set_ylabel("Frequency (Hz)")
axs[4].grid(True)

# גרף 6: Spectral Roll-off
axs[5].plot(librosa_times, rolloff[0], color='purple', linewidth=2)
axs[5].set_title("6. Spectral Roll-off (85% Energy Cutoff Frequency)", fontsize=12, fontweight='bold')
axs[5].set_ylabel("Frequency (Hz)")
axs[5].set_xlabel("Time (Seconds)")
axs[5].grid(True)

# התאמת עיצוב סופית ושמירה
plt.tight_layout()
plt.savefig("voice_analysis_report.png", dpi=300) # שומר את הגרף המשולב כתמונה איכותית לדוח
plt.show()
