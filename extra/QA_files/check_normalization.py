import os
import glob
import librosa
import numpy as np
from pathlib import Path

# =====================================================================
# הגדרת נתיבי התיקיות המנורמלות לבדיקה
# =====================================================================
# שנה את הנתיבים במידת הצורך למיקום שבו חילצת את ה-ZIP במחשב שלך
TARGET_DIRS = [
    Path("../../sentences_samples_normalized"),
    Path("../../audio_by_accent_raw_normalized")
]

# סף כשל מרבי מותר
MAX_ALLOWED_FAILURES = 5

# פרמטרי הבדיקה (תואמים בדיוק ל-audio_normalization.py)
TARGET_SR = 16000
MIN_DURATION_SEC = 0.1
EXPECTED_PEAK_LOWER = 0.90  # גבול תחתון סביר ל-Peak (עבור אותות שהיו כמעט שקטים)
EXPECTED_PEAK_UPPER = 0.96  # גבול עליון (הנורמליזציה מכוונת ל-0.95)

print("🔍 מתחיל בדיקת תקינות נורמליזציה עבור כל הקבצים...")

failed_files = []
total_files_checked = 0

for target_dir in TARGET_DIRS:
    if not target_dir.exists():
        print(f"⚠️  התיקייה לא נמצאה, מדלג: {target_dir}")
        continue

    # סריקה ריקורסיבית של כל קובצי ה-WAV
    wav_files = list(target_dir.rglob("*.wav"))
    print(f"📁 בודק {len(wav_files)} קבצים בתיקייה: {target_dir.name}...")

    for file_path in wav_files:
        total_files_checked += 1
        try:
            # 1. טעינת קובץ השמע
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

            # בדיקת קובץ ריק
            if len(y) == 0:
                failed_files.append((str(file_path), "קובץ שמע ריק"))
                continue

            # 2. בדיקת אורך מינימלי (חייב להיות >= 0.1 שניות)
            duration_sec = len(y) / sr
            if duration_sec < MIN_DURATION_SEC:
                failed_files.append((str(file_path), f"קובץ קצר מדי: {duration_sec:.3f} שניות"))
                continue

            # 3. בדיקת Peak Normalization (עוצמת שיא בסביבות 0.95)
            max_peak = np.max(np.abs(y))
            if max_peak > EXPECTED_PEAK_UPPER or max_peak < EXPECTED_PEAK_LOWER:
                failed_files.append((str(file_path), f"Peak Amplitude לא תקין: {max_peak:.4f}"))
                continue

        except Exception as e:
            failed_files.append((str(file_path), f"שגיאה בטעינת הקובץ: {str(e)}"))

# =====================================================================
# סיכום והקפצת שגיאה במידת הצורך
# =====================================================================
failed_count = len(failed_files)

print("\n" + "=" * 70)
print(f"📊 סה\"כ קבצים שנבדקו: {total_files_checked}")
print(f"❌ סה\"כ קבצים שנכשלו בבדיקה: {failed_count}")
print("=" * 70)

if failed_count > 0:
    print("\nפירוט קבצים שנכשלו:")
    for path, reason in failed_files[:10]:  # מדפיס עד 10 קבצים לראות דוגמאות
        print(f"  • {path} --> {reason}")
    if failed_count > 10:
        print(f"  ... ועוד {failed_count - 10} קבצים.")

# בדיקת התנאי: הקפצת שגיאה אם יש יותר מ-5 קבצים תקולים
if failed_count > MAX_ALLOWED_FAILURES:
    raise ValueError(
        f"\n❌ [FAILURE]: בדיקת הנורמליזציה נכשלה! "
        f"נמצאו {failed_count} קבצים שאינם מנורמלים כראוי (הסף המרבי המותר הוא {MAX_ALLOWED_FAILURES})."
    )
else:
    print(
        f"\n🎉 [SUCCESS]: הבדיקה עברה בהצלחה! (נמצאו {failed_count} חריגות, שזה בדיוק בסף המותר של עד {MAX_ALLOWED_FAILURES}).")