import os
import shutil
import pandas as pd
from pathlib import Path


def merge_dataset_clean():
    current_dir = Path(__file__).resolve().parent
    source_base = current_dir / "split_train_and_test_by_accent"
    output_base = current_dir / "free_text_train_and_test"

    splits = ['train', 'validation', 'test']

    if not source_base.exists():
        print(f"Error: Source directory {source_base} does not exist.")
        return

    # מילון שיחזיק את ה-DataFrames המאוחדים לכל split
    split_frames = {split: [] for split in splits}

    print("Starting clean dataset unification (dropping age and gender)...")

    # עוברים על כל תיקיות המבטאים
    for accent_folder in os.listdir(source_base):
        accent_dir = source_base / accent_folder
        if not accent_dir.is_dir():
            continue

        print(f"\n[Processing Accent] '{accent_folder}'")

        for split in splits:
            split_dir = accent_dir / split
            if not split_dir.exists():
                continue

            # איתור קובץ המטא-דאטה
            meta_file_path = None
            for file in os.listdir(split_dir):
                if file.endswith(('.csv', '.xlsx', '.xls')):
                    meta_file_path = split_dir / file
                    break

            if not meta_file_path or not meta_file_path.exists():
                continue

            # קריאה בטוחה של הקובץ
            try:
                if meta_file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(meta_file_path)
                else:
                    df = pd.read_csv(meta_file_path)

                df.columns = df.columns.str.strip().str.lower()
            except Exception as e:
                print(f" -> Error reading {meta_file_path.name}: {e}")
                continue

            df_cleaned = pd.DataFrame()

            # פונקציית עזר לבחירת העמודה הקיימת הראשונה מתוך רשימת אפשרויות
            def get_first_existing_column(dataframe, alternative_keys):
                for key in alternative_keys:
                    if key in dataframe.columns:
                        return dataframe[key]
                return None

            # מיפוי ארבעת עמודות הליבה בלבד
            if accent_folder == 'chinese':
                df_cleaned['filename'] = get_first_existing_column(df, ['wav_path', 'filename', 'file', 'audio'])
                df_cleaned['speaker_id'] = get_first_existing_column(df,
                                                                     ['speaker_id', 'speaker-id', 'id', 'client_id'])
                df_cleaned['accent'] = 'chinese'
                df_cleaned['text'] = get_first_existing_column(df, ['transcription', 'text', 'sentence'])

            elif accent_folder in ['indian', 'nigerian']:
                df_cleaned['filename'] = get_first_existing_column(df, ['audio', 'filename', 'file', 'wav_path'])
                df_cleaned['speaker_id'] = get_first_existing_column(df, ['client_id', 'speaker_id', 'id'])
                df_cleaned['accent'] = get_first_existing_column(df,
                                                                 ['accent']) if 'accent' in df.columns else accent_folder
                df_cleaned['text'] = get_first_existing_column(df, ['sentence', 'text', 'transcription'])

            elif accent_folder == 'usa':
                df_cleaned['filename'] = get_first_existing_column(df, ['file', 'filename', 'audio', 'wav_path'])
                df_cleaned['speaker_id'] = get_first_existing_column(df,
                                                                     ['speaker-id', 'id', 'speaker_id', 'client_id'])
                df_cleaned['accent'] = 'usa'
                df_cleaned['text'] = get_first_existing_column(df, ['text', 'sentence', 'transcription'])

            # סינון שורות שאין בהן שם קובץ תקני
            if 'filename' in df_cleaned.columns and df_cleaned['filename'] is not None:
                df_cleaned = df_cleaned.dropna(subset=['filename'])
            else:
                continue

            # יצירת תיקיית היעד והעתקת קובצי האודיו הפיזיים
            dest_split_dir = output_base / split
            os.makedirs(dest_split_dir, exist_ok=True)

            for original_filename in df_cleaned['filename']:
                pure_name = os.path.basename(str(original_filename))
                src_file = split_dir / pure_name

                if src_file.exists():
                    shutil.copy2(src_file, dest_split_dir / pure_name)
                else:
                    # חיפוש גמיש (Case-Insensitive) למקרה של סיומות כמו .WAV גדולות
                    for f in os.listdir(split_dir):
                        if f.lower() == pure_name.lower():
                            shutil.copy2(split_dir / f, dest_split_dir / f)
                            break

            # ניקוי נתיבים משם הקובץ בטבלה הסופית
            df_cleaned['filename'] = df_cleaned['filename'].apply(
                lambda x: os.path.basename(str(x)) if pd.notna(x) else x)

            # ניקוי והחלקה של ה-speaker_id שיוצג בצורה אחידה כטקסט נקי
            df_cleaned['speaker_id'] = df_cleaned['speaker_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) and (
                        isinstance(x, float) or (isinstance(x, str) and '.' in str(x))) else (
                str(x).strip() if pd.notna(x) else x))

            split_frames[split].append(df_cleaned)
            print(f" -> Merged {len(df_cleaned)} rows for {accent_folder} ({split})")

    # כתיבת קובצי המטא-דאטה המאוחדים הסופיים
    print("\nWriting final unified metadata sheets...")
    for split in splits:
        if split_frames[split]:
            df_final = pd.concat(split_frames[split], ignore_index=True)

            # סדר עמודות אחיד וסופי
            column_order = ['filename', 'speaker_id', 'accent', 'text']
            df_final = df_final[column_order]

            output_meta_path = output_base / split / f"{split}_metadata.csv"
            df_final.to_csv(output_meta_path, index=False, encoding='utf-8-sig')
            print(f" -> Successfully created: {output_meta_path.relative_to(current_dir)} ({len(df_final)} total rows)")


if __name__ == "__main__":
    print("=============================================")
    print("Initializing Core-Feature Dataset Unifier... ")
    print("=============================================")
    merge_dataset_clean()
    print("\nProcess finished! Unified dataset is ready in 'free_text_train_and_test'.")