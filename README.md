# AI Accent Recognition

פרויקט לסיווג מבטאים באנגלית לארבע קבוצות.

```text
chinese
indian / hindi
nigerian
usa
```

בפרויקט נבדקו שלושה מסלולי ניסוי עיקריים: מסווגי למידת מכונה על פיצ'רים שחולצו מהאודיו, מודל CNN אקוסטי המבוסס על ספקטרוגרמות, ומודל Multimodal CNN המשלב ספקטרוגרמה עם פיצ'רים פונטיים.

## מבנה הנתונים

נתוני הדיבור החופשי מרוכזים בתיקייה:

```text
audio_by_accent_raw/
├── chinese/
├── indian/
├── nigerian/
└── usa/
```

כל תיקייה מכילה את קובצי האודיו של המבטא המתאים לאחר איחוד המקורות.

נתוני Speech Accent Archive נמצאים בתיקייה:

```text
accents_audio/
└── accent_archive/
    ├── chinese/
    ├── hindi/
    ├── nigerian/
    └── usa/
```

גם כאן קובצי האודיו מסודרים לפי מבטא.

נתוני Speech Accent Archive נחתכים בהמשך לששת המשפטים בעזרת הקבצים שבתיקיית:

```text
cut_and_normalize/
```

הפלט נשמר במבנה:

```text
sentences_samples/
├── chinese/
│   ├── sentence_0/
│   ├── sentence_1/
│   └── ...
├── indian/
├── nigerian/
└── usa/
```

## הכנת מאגרי הדיבור החופשי

הנתונים המקוריים נמצאים תחת:

```text
all_accents_data/
```

חלקם הורדו ישירות ממאגרים חיצוניים וחלקם נוצרו בעזרת הקבצים שבתיקיית:

```text
export_data/
```

הקובץ הבא מאחד את המקורות לפי מבטא ובהתאם לכמויות שנבחרו:

```text
group_sources_to_one_by_accent.py
```

הפלט שלו הוא:

```text
audio_by_accent_raw/
```

## יצירת Metadata

הקובץ:

```text
metadata_generator.py
```

יוצר קובצי metadata לשלושת מצבי העבודה:

```text
archive_only
free_text
mixed
```

הקבצים נשמרים בתיקיות:

```text
machine_learning_official_archive_only/
machine_learning_official_free_text_only/
machine_learning_official_mixed/
```

## יצירת פיצ'רים - שלב משותף

לאחר הכנת הנתונים, שני ענפי הלמידה עוברים דרך:

```text
create_features.py
```

הקוד משתמש במימושים שבתיקיית:

```text
features/
```

ויוצר את הקבצים:

```text
master_extracted_features_archive_only.csv
master_extracted_features_free_text.csv
master_extracted_features_mixed.csv
```

לאחר שלב זה התהליך מתפצל למסלול Machine Learning ולמסלולי CNN.

## מסלול Machine Learning

קובצי האודיו המשמשים לחילוץ הפיצ'רים עוברים נרמול באמצעות:

```text
cut_and_normalize/normalize_all_for_2_folders.py
```

הגרסאות המנורמלות נשמרות תחת:

```text
audio_by_accent_raw_normalized/
sentences_samples_normalized/
```

הניסויים מורצים באמצעות:

```text
main_for_machine_learning.py
```

והמסווגים נמצאים בתיקיית:

```text
classifiers/
```

הגדרות מרכזיות:

```python
RUN_MODE = "free_text"
ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 30
TUNING_MODE = "random_search"
```

## מסלול CNN

לאחר יצירת הפיצ'רים, הספקטרוגרמות נוצרות באמצעות:

```text
cut_and_normalize/spectogram_creation.py
```

הן נשמרות כקובצי PyTorch בתיקייה המתאימה למצב ההרצה:

```text
spectogram_logmel_<mode>/
```

מודל ה-CNN האקוסטי מורץ באמצעות:

```text
CNN.py
```

המודל המולטי-מודאלי מורץ באמצעות:

```text
Multimodal_CNN.py
```

המודל המולטי-מודאלי משתמש גם בספקטרוגרמות וגם בפיצ'רים שנוצרו קודם לכן.

## Kaggle

מומלץ להריץ את שלוש מחברות הניסוי ב-Kaggle בגלל זמני הריצה ומשאבי החישוב.

```text
Machine Learning notebook: [להוסיף קישור]
CNN notebook:              [להוסיף קישור]
Multimodal CNN notebook:   [להוסיף קישור]
```

לכל מחברת יש לצרף את ה-Dataset המתאים לפני ההרצה.

## תוצאות וגרפים

חלק מהגרפים נוצרים במהלך ההרצה. ב-Kaggle הם מוצגים בפלט של תאי המחברת. בהרצה מקומית ב-PyCharm הם נשמרים גם בתיקייה:

```text
graphs/
```

תוצאות נוספות נשמרות תחת:

```text
results/
```

## מסמכי ההגשה

בתיקייה הראשית נמצאים:

```text
README.md
libs.txt
EXPERIMENT_REPRODUCTION_GUIDE.docx
```

שאר מסמכי התיעוד נמצאים תחת:

```text
documentation/
```

ובתוכה מדריך למשתמש ותיעוד חיצוני למתכנת, בגרסאות Word ו-Markdown.

## מקורות הנתונים

מאגרי הנתונים נלקחו ממקורות חיצוניים, ובהם Speech Accent Archive, Kaggle, Hugging Face, OpenSLR ו-Mozilla Common Voice. פירוט מלא של המקורות והקישורים מופיע בביבליוגרפיה של דו"ח הפרויקט.
