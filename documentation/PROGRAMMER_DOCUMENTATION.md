# תיעוד חיצוני למתכנת

## 1. מבנה כללי

המערכת מתחילה ממאגרי אודיו ממספר מקורות. לאחר ארגון הנתונים, חיתוך ההקלטות ויצירת metadata, מופעל שלב משותף של חילוץ פיצ'רים. לאחר מכן התהליך מתפצל למסלול Machine Learning ולמסלולי CNN.

```text
External datasets
        |
        +-----------------------------+
        |                             |
all_accents_data/          accents_audio/accent_archive/
        |                             |
group_sources_to_one_       cut_and_normalize/
by_accent.py                           |
        |                             |
audio_by_accent_raw/        sentences_samples/
        +-------------+---------------+
                      |
             metadata_generator.py
                      |
              create_features.py
                      |
       master_extracted_features_<mode>.csv
                      |
          +-----------+--------------------+
          |                                |
 Machine Learning                     Deep Learning
          |                                |
main_for_machine_learning.py          spectogram_creation.py
          |                                |
     classifiers/                  spectogram_logmel_<mode>/
                                           |
                                  +--------+---------+
                                  |                  |
                                CNN.py       Multimodal_CNN.py
```

## 2. מבנה קובצי האודיו

לאחר איחוד מקורות ה-Free Text, קובצי האודיו מסודרים לפי מבטא:

```text
audio_by_accent_raw/
├── chinese/
├── indian/
├── nigerian/
└── usa/
```

Speech Accent Archive נשמר בנפרד וגם הוא מסודר לפי מבטא:

```text
accents_audio/
└── accent_archive/
    ├── chinese/
    ├── hindi/
    ├── nigerian/
    └── usa/
```

ההפרדה לפי מבטא נשמרת גם בשלבי העיבוד הבאים כדי לאפשר יצירת metadata ובקרה על הרכב הנתונים.

## 3. מצבי הנתונים

המערכת תומכת בשלושה מצבים:

```text
archive_only
free_text
mixed
```

אותו mode צריך להישמר בין ה-metadata, קובץ הפיצ'רים, הספקטרוגרמות וקובץ ההרצה.

## 4. Metadata ופיצ'רים

`metadata_generator.py` יוצר את קובצי ה-metadata.

`create_features.py` מפעיל את חילוץ הפיצ'רים שבתיקיית `features/` ויוצר:

```text
master_extracted_features_archive_only.csv
master_extracted_features_free_text.csv
master_extracted_features_mixed.csv
```

זהו שלב משותף לפני הפיצול. מסלול Machine Learning משתמש בטבלת הפיצ'רים כקלט המרכזי, ו-Multimodal CNN משתמש בחלק מהפיצ'רים יחד עם הספקטרוגרמה.

## 5. מסלול למידת מכונה

הקובץ:

```text
main_for_machine_learning.py
```

מנהל את ניסויי למידת המכונה. המסווגים נמצאים תחת:

```text
classifiers/
```

הקלט המרכזי הוא קובץ:

```text
master_extracted_features_<mode>.csv
```

הקוד מאפשר לבחור את מצב הנתונים, להפעיל Feature Selection, לקבוע Top K ולבחור את שיטת כוונון ההיפר-פרמטרים.

## 6. המודל האקוסטי

הספקטרוגרמות נוצרות באמצעות:

```text
cut_and_normalize/spectogram_creation.py
```

ונשמרות תחת:

```text
spectogram_logmel_<mode>/
```

המודל האקוסטי מופעל באמצעות:

```text
CNN.py
```

## 7. המודל המולטי-מודאלי

המודל המולטי-מודאלי מופעל באמצעות:

```text
Multimodal_CNN.py
```

הוא מקבל שני סוגי קלט: ספקטרוגרמת Log-Mel ופיצ'רים פונטיים מתוך קובץ הפיצ'רים.

## 8. מחברת למידת המכונה

המחברת בנויה ממספר חלקים עיקריים:

- בתחילת המחברת מוגדר מצב ההרצה, מאותרים קובצי הפיצ'רים ונבדקת תקינות הנתונים.
- לאחר מכן מתבצע preprocessing של הטבלה: הסרת עמודות שאינן פיצ'רים, טיפול בערכים חסרים ונרמול.
- אם Feature Selection מופעל, הפיצ'רים מדורגים באמצעות ANOVA ונבחרים הפיצ'רים המובילים.
- האימון מתבצע ב-5-Fold Group Cross-Validation לפי דובר, ולאחריו נבדק שאין Speaker Leakage.
- בסיום מוצגים Accuracy, Macro F1, מטריצות בלבול וניתוחים של חשיבות ותרומת הפיצ'רים.

מחברת:

```text
[להוסיף קישור]
```

## 9. מחברת המודל האקוסטי

המחברת מחולקת לשלבים הבאים:

- הגדרת מצב ההרצה, בחירת GPU או CPU, איתור ה-metadata והספקטרוגרמות ובדיקת התאמה ביניהם.
- הגדרת Dataset, SpecAugment ומודל CNN בעל שלושה בלוקים קונבולוציוניים.
- אימון והערכה ב-5-Fold Group Cross-Validation לפי דובר, כולל Validation ושמירת המודל הטוב בכל Fold.
- בדיקה מפורשת שאין חפיפה של דוברים בין Train, Validation ו-Test.
- הפקת מדדי Accuracy ו-Macro F1, מטריצות בלבול וגרפים לניתוח המודל, כגון Grad-CAM, ספקטרוגרמות ממוצעות ו-Spectral Rolloff.

מחברת:

```text
[להוסיף קישור]
```

## 10. מחברת המודל המולטי-מודאלי

המחברת דומה למסלול ה-CNN, אך מוסיפה זרוע פונטית:

- טעינת הספקטרוגרמה יחד עם וקטור פונטי הכולל חמישה פיצ'רים.
- הזרוע האקוסטית מעבדת את הספקטרוגרמה והזרוע הפונטית מעבדת את הווקטור.
- שני הייצוגים משולבים לפני הסיווג באמצעות מנגנון Gated Residual Fusion.
- האימון מתבצע ב-5-Fold Group Cross-Validation עם בדיקת Speaker Leakage ושמירת המודל הטוב.
- בסיום מופקים מדדי ביצוע, מטריצות בלבול וגרפי ניתוח. המחברת כוללת גם מנגנון אינפרנס לקובץ אודיו חדש, הכולל תמלול, יצירת הווקטור הפונטי, יצירת הספקטרוגרמה וסיווג המבטא.

מחברת:

```text
[להוסיף קישור]
```

## 11. גרפים ותוצרים

ב-Kaggle חלק מהגרפים מוצגים ישירות בפלט המחברת. בהרצה מקומית ב-PyCharm הם נשמרים גם תחת:

```text
graphs/
```

תוצאות נוספות נשמרות תחת:

```text
results/
```

## 12. הוספת פיצ'ר חדש

כדי להוסיף פיצ'ר חדש יש לממש אותו תחת `features/`, לחבר אותו ל-`create_features.py` ולהוסיף אותו לטבלת הפיצ'רים. אם מופעל Feature Selection, יש לוודא שהבחירה מתבצעת רק על נתוני האימון.

## 13. הוספת מסווג חדש

מסווג Machine Learning חדש צריך להיכנס לתיקיית `classifiers/` ולהתחבר ל-`main_for_machine_learning.py`. רצוי לשמור על אותו ממשק קלט ופלט של המסווגים הקיימים.

## 14. קלט ופלט

מסלול Machine Learning:

```text
audio
-> normalization
-> create_features.py
-> master_extracted_features_<mode>.csv
-> main_for_machine_learning.py
-> results / graphs
```

מסלול CNN:

```text
audio
-> create_features.py
-> spectogram_creation.py
-> spectogram_logmel_<mode>/
-> CNN.py
-> results / graphs
```

מסלול Multimodal CNN:

```text
audio
-> create_features.py
-> master_extracted_features_<mode>.csv
-> spectogram_creation.py
-> spectogram_logmel_<mode>/
-> Multimodal_CNN.py
-> results / graphs
```
