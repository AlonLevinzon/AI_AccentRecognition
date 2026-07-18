# import os
# import csv
# import io
# import soundfile as sf
# from datasets import load_dataset, Audio
#
# # 1. טעינת ה-Dataset מהאתר
# print("מוריד את ה-Dataset...")
# dataset = load_dataset("CAiRE/ASCEND", split="train")
#
# # שינוי קריטי: ביטול הפיענוח של Hugging Face כדי לעקוף את השגיאה של torchcodec
# dataset = dataset.cast_column("audio", Audio(decode=False))
#
# # 2. יצירת תיקייה לשמירת קבצי האודיו והמידע
# output_dir = "chinese_english_audio_clips"
# os.makedirs(output_dir, exist_ok=True)
#
# metadata_file_path = os.path.join(output_dir, "metadata.csv")
#
# # 3. סינון ושמירה של קטעי האודיו וכל המידע הזמין
# print("מסנן ושומר קטעי אודיו ומידע מלא באנגלית בלבד...")
#
# with open(metadata_file_path, "w", newline="", encoding="utf-8") as csvfile:
#     # הגדרת כל השדות הזמינים במאגר על הדוברים והטקסט
#     fieldnames = ["file_name", "id", "speaker_id", "gender", "duration", "language", "transcription"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for item in dataset:
#         if item["language"] == "en":
#             file_id = item["id"]
#             file_name = f"clip_{file_id}.wav"
#             file_path = os.path.join(output_dir, file_name)
#
#             try:
#                 # א) חילוץ ידני של האודיו מהביטים הגולמיים (מונע לחלוטין את השימוש ב-torchcodec)
#                 audio_bytes = item["audio"]["bytes"]
#                 audio_buffer = io.BytesIO(audio_bytes)
#
#                 # קריאת המערך וקצב הדגימה ישירות מהבאפר בזיכרון
#                 audio_array, sampling_rate = sf.read(audio_buffer)
#
#                 # שמירת קובץ ה-WAV התקני למחשב
#                 sf.write(file_path, audio_array, sampling_rate)
#
#                 # ב) חילוץ כל שאר פרטי הדובר והטקסט שלו
#                 row_data = {
#                     "file_name": file_name,
#                     "id": file_id,
#                     "speaker_id": item.get("speaker_id", ""),
#                     "gender": item.get("gender", ""),
#                     "duration": item.get("duration", ""),
#                     "language": item.get("language", ""),
#                     "transcription": item.get("transcription", "").strip()
#                 }
#
#                 # כתיבת המידע לקובץ ה-CSV
#                 writer.writerow(row_data)
#
#             except Exception as e:
#                 print(f"שגיאה בעיבוד פריט {file_id}: {e}")
#                 continue
#
# print(f"\nהסתיים בהצלחה!")
# print(f"1. קבצי האודיו התקינים נשמרו בתיקייה: {output_dir}")
# print(f"2. קובץ ה-CSV עם כל המידע והטקסט נשמר ב: {metadata_file_path}")
#

#for the multi accents from load_dataset("westbrook/English_Accent_DataSet")
# import os
# import csv
# import io
# import soundfile as sf
# from datasets import load_dataset, Audio
#
# print("Loading dataset...")
# dataset = load_dataset("westbrook/English_Accent_DataSet", split="train")
#
# dataset = dataset.cast_column("audio", Audio(decode=False))
#
# output_dir = "multi_english_audio_clips"
# os.makedirs(output_dir, exist_ok=True)
#
# metadata_file_path = os.path.join(output_dir, "metadata.csv")
#
# fieldnames = ["file_name", "audio_id", "speaker_id", "gender", "duration", "accent", "raw_text"]
#
# with open(metadata_file_path, "w", newline="", encoding="utf-8") as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     print("Extracting audio files and metadata...")
#     for item in dataset:
#         file_id = item.get("audio_id", "unknown")
#         file_name = f"clip_{file_id}.wav"
#         file_path = os.path.join(output_dir, file_name)
#
#         try:
#             audio_bytes = item["audio"]["bytes"]
#             audio_buffer = io.BytesIO(audio_bytes)
#             audio_array, sampling_rate = sf.read(audio_buffer)
#
#             sf.write(file_path, audio_array, sampling_rate)
#
#             row_data = {
#                 "file_name": file_name,
#                 "audio_id": file_id,
#                 "speaker_id": item.get("speaker_id", ""),
#                 "gender": item.get("gender", ""),
#                 "duration": item.get("duration", ""),
#                 "accent": item.get("accent", ""),
#                 "raw_text": item.get("raw_text", "").strip()
#             }
#
#             writer.writerow(row_data)
#
#         except Exception as e:
#             print(f"Error processing item {file_id}: {e}")
#             continue
#
# print(f"\nExtraction completed successfully!")
# print(f"1. Audio files saved inside: {output_dir}")
# print(f"2. Metadata CSV saved at: {metadata_file_path}")
#
# for validaion and test group which in westbrook/English_Accent_DataSet on huggingface
# import os
# import csv
# import io
# import soundfile as sf
# from datasets import load_dataset, Audio
#
#
# base_output_dir = "multi_english_dataset_split"
# os.makedirs(base_output_dir, exist_ok=True)
#
# fieldnames = ["file_name", "audio_id", "speaker_id", "gender", "duration", "accent", "raw_text"]
#
# splits_to_extract = ["validation", "test"]
#
# for split_name in splits_to_extract:
#     print(f"\nLoading and preparing {split_name} split...")
#     dataset = load_dataset("westbrook/English_Accent_DataSet", split=split_name)
#
#     dataset = dataset.cast_column("audio", Audio(decode=False))
#
#     split_dir = os.path.join(base_output_dir, split_name)
#     os.makedirs(split_dir, exist_ok=True)
#
#     metadata_file_path = os.path.join(split_dir, f"metadata_{split_name}.csv")
#
#     with open(metadata_file_path, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#
#         print(f"Extracting audio files and metadata for {split_name}...")
#         for item in dataset:
#             file_id = item.get("audio_id", "unknown")
#             file_name = f"clip_{file_id}.wav"
#             file_path = os.path.join(split_dir, file_name)
#
#             try:
#                 audio_bytes = item["audio"]["bytes"]
#                 audio_buffer = io.BytesIO(audio_bytes)
#                 audio_array, sampling_rate = sf.read(audio_buffer)
#
#                 sf.write(file_path, audio_array, sampling_rate)
#
#                 row_data = {
#                     "file_name": file_name,
#                     "audio_id": file_id,
#                     "speaker_id": item.get("speaker_id", ""),
#                     "gender": item.get("gender", ""),
#                     "duration": item.get("duration", ""),
#                     "accent": item.get("accent", ""),
#                     "raw_text": item.get("raw_text", "").strip()
#                 }
#
#                 writer.writerow(row_data)
#
#             except Exception as e:
#                 print(f"Error processing item {file_id} in {split_name}: {e}")
#                 continue
#
# print(f"\nExtraction completed successfully!")
# print(f"Validation and Test splits are saved separately inside: {base_output_dir}")
# #
# from datasets import load_dataset
#
# # Load the dataset structure
# dataset = load_dataset("westbrook/English_Accent_DataSet", split="train")
#
# # Get the list of names corresponding to the numbers
# accent_names = dataset.features["accent"].names
#
# # Print the mapping
# print("Accent Mapping:")
# for index, name in enumerate(accent_names):
#     print(f"Number {index}: {name}")

import os
import io
import csv
import soundfile as sf
from datasets import load_dataset, Audio

# הגדרת ה-Cache של Hugging Face לכונן D
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache"

print("Loading HamdanXI/speech-accent-archive-v2 (train split)...")
try:
    dataset = load_dataset("HamdanXI/speech-accent-archive-v2", split="train")
except Exception:
    dataset = load_dataset("HamdanXI/speech-accent-archive-v2")
    if isinstance(dataset, dict):
        dataset = dataset["train"]

# הגדרת פענוח עצלן (Lazy Decoding) לשמירה על זיכרון ה-RAM
dataset = dataset.cast_column("audio", Audio(decode=False))

# הגדרת תיקיית היעד המשותפת שלך
output_dir = "more_accents_data"
os.makedirs(output_dir, exist_ok=True)

# קובץ מטא-דאטה ייעודי עבור התוספת מהמאגר הזה
metadata_file_path = os.path.join(output_dir, "metadata_hamdan_raw.csv")
fieldnames = ["file_name", "speaker_id", "text", "gender", "accent"]

# מיפוי המבטאים שאנחנו מחפשים לערכים הרשמיים שנכתוב ב-CSV
accent_mapping = {
    "spanish": "Spanish",
    "spanish": "Spanish",
    "french": "French",
    "french": "French",
    "german": "German",
    "german": "German"
}

# מוני קבצים נפרדים לכל מבטא בשביל שמות הקבצים
counters = {"Spanish": 0, "French": 0, "German": 0}
total_extracted = 0

print(f"Extracting Spanish, French, and German files into '{output_dir}' folder...")
with open(metadata_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item in dataset:
        # חילוץ ונרמול תגית המבטא מהשורה הנוכחית
        raw_accent = str(item.get("accent", "")).strip().lower()

        # אם המבטא הנוכחי הוא אחד משלושת אלו שאנחנו מחפשים
        if raw_accent in accent_mapping:
            normalized_accent = accent_mapping[raw_accent]
            current_id = counters[normalized_accent]

            # יצירת שם קובץ ייחודי (למשל: hamdan_french_0.wav)
            file_name = f"hamdan_{normalized_accent.lower()}_{current_id}.wav"
            file_path = os.path.join(output_dir, file_name)

            try:
                # חילוץ הביטים והמרתם לקובץ WAV פיזי
                audio_bytes = item["audio"]["bytes"]
                audio_buffer = io.BytesIO(audio_bytes)
                audio_array, sampling_rate = sf.read(audio_buffer)

                sf.write(file_path, audio_array, sampling_rate)

                # שמירת הנתונים במטא-דאטה
                writer.writerow({
                    "file_name": file_name,
                    "speaker_id": item.get("speaker_id", item.get("client_id", "")),
                    "text": item.get("text", item.get("sentence", "")).strip(),
                    "gender": item.get("gender", ""),
                    "accent": normalized_accent
                })

                counters[normalized_accent] += 1
                total_extracted += 1

            except Exception as e:
                # דילוג במקרה של קובץ שמע פגום
                continue

print(f"\nExtraction completed successfully!")
print(f"Saved under '{output_dir}/':")
for accent_type, count in counters.items():
    print(f"- {accent_type}: {count} files")
print(f"Total new files added: {total_extracted}")
print(f"Metadata saved at: {metadata_file_path}")