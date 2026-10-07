# מדריך למשתמש

המדריך מתאר את סדר ההפעלה של הפרויקט. אם קובצי הביניים כבר קיימים, אפשר להתחיל מהשלב הרלוונטי ואין צורך לייצר את כל הנתונים מחדש.

## 1. סביבת העבודה

מאגר הפרויקט נמצא ב-GitHub.

```text
https://github.com/AlonLevinzon/AI_AccentRecognition.git
```

רשימת הספריות הדרושות להרצה מופיעה בקובץ:

```text
libs.txt
```

## 2. עבודה בסביבת הענן

לניסויים הכבדים מומלץ להשתמש ב-Kaggle. יש ליצור חשבון או להיכנס לחשבון קיים, לפתוח את קישור המחברת המתאים ולצרף אליה את ה-Dataset שנדרש להרצה.

Kaggle מספק מכסת GPU חינמית של עד כ-30 שעות בשבוע. המכסה והזמינות יכולות להשתנות, ולכן כדאי לבדוק את ה-quota שמופיע בחשבון לפני הרצה ארוכה.

כדי להפעיל GPU במחברת:

1. פותחים את המחברת במצב Edit.
2. בצד ימין פותחים את Settings.
3. תחת Accelerator בוחרים GPU.
4. ממתינים להפעלה מחדש של ה-session.
5. מריצים את המחברת.

במחברות PyTorch הקוד בודק אם GPU זמין ומשתמש בו בהתאם. הפעלת Accelerator בלבד אינה מעבירה קוד אוטומטית ל-GPU; המודל והטנזורים צריכים להישלח ל-device המתאים, כפי שנעשה בקוד הפרויקט.

לפני כל הרצה חשוב לוודא שה-Dataset המתאים מופיע ברשימת הקלטים של המחברת.

קישורי המחברות:

```text
Machine Learning notebook: [להוסיף קישור]
CNN notebook:              [להוסיף קישור]
Multimodal CNN notebook:   [להוסיף קישור]
```

## 3. מבנה קובצי האודיו

נתוני הדיבור החופשי לאחר איחוד המקורות נמצאים במבנה:

```text
audio_by_accent_raw/
├── chinese/
├── indian/
├── nigerian/
└── usa/
```

נתוני Speech Accent Archive נמצאים במבנה:

```text
accents_audio/
└── accent_archive/
    ├── chinese/
    ├── hindi/
    ├── nigerian/
    └── usa/
```

כל אחת מהתיקיות מכילה את קובצי האודיו של המבטא המתאים.

## 4. הכנת מאגרי הדיבור החופשי

הנתונים המקוריים נמצאים תחת:

```text
all_accents_data/
```

כדי לאחד את המקורות לפי מבטא מריצים:

```bash
python group_sources_to_one_by_accent.py
```

הפלט נשמר תחת:

```text
audio_by_accent_raw/
```

## 5. חיתוך מאגר המבטאים

החיתוך מתבצע בעזרת הקבצים שבתיקיית:

```text
cut_and_normalize/
```

הקבצים המרכזיים הם:

```text
audio_time_split.py
sentences_extractor.py
cut_archive_to_sentences.py
```

התהליך נעזר ב-Whisper כדי לחלק את ההקלטה לששת המשפטים. הפלט נשמר תחת:

```text
sentences_samples/
```

ובתוך כל מבטא קיימת תיקייה נפרדת לכל משפט.

## 6. יצירת קובצי המטא-דאטה

מריצים:

```bash
python metadata_generator.py
```

נוצרים קובצי metadata עבור:

```text
archive_only
free_text
mixed
```

## 7. נרמול ויצירת פיצ'רים

למסלול הפיצ'רים מריצים תחילה:

```text
cut_and_normalize/normalize_all_for_2_folders.py
```

הפלט נשמר ב:

```text
audio_by_accent_raw_normalized/
sentences_samples_normalized/
```

לאחר מכן מריצים את השלב המשותף:

```bash
python create_features.py
```

נוצרים:

```text
master_extracted_features_archive_only.csv
master_extracted_features_free_text.csv
master_extracted_features_mixed.csv
```

קבצים אלה משמשים את מסלול Machine Learning, והפיצ'רים הרלוונטיים משמשים גם את Multimodal CNN.

## 8. הרצת מסווגי למידת מכונה

פותחים את:

```text
main_for_machine_learning.py
```

ובוחרים את מצב הנתונים:

```python
RUN_MODE = "free_text"
```

אפשר להפעיל בחירת פיצ'רים ולקבוע את מספר הפיצ'רים:

```python
ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 30
```

מצב כוונון ההיפר-פרמטרים נקבע באמצעות:

```python
TUNING_MODE = "random_search"
```

לאחר בחירת ההגדרות מריצים את הקובץ.

## 9. יצירת ספקטרוגרמות

מריצים:

```text
cut_and_normalize/spectogram_creation.py
```

הפלט נשמר בתיקייה המתאימה למצב הנתונים:

```text
spectogram_logmel_<mode>/
```

## 10. הרצת המודל האקוסטי

למודל האקוסטי מריצים:

```bash
python CNN.py
```

המודל משתמש בספקטרוגרמות של מצב הנתונים שנבחר.

## 11. הרצת המודל המולטי-מודאלי

למודל המולטי-מודאלי מריצים:

```bash
python Multimodal_CNN.py
```

לפני ההרצה יש לוודא שקיימים גם קובצי הספקטרוגרמות וגם קובץ הפיצ'רים המתאים לאותו מצב.

## 12. תוצאות וגרפים

ב-Kaggle חלק מהגרפים מוצגים ישירות בפלט של המחברת.

בהרצה מקומית ב-PyCharm הגרפים נשמרים גם תחת:

```text
graphs/
```

תוצאות נוספות נשמרות תחת:

```text
results/
```
