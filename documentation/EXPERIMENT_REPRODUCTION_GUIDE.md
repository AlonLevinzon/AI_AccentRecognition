# מדריך לבודק - שחזור הניסויים

המדריך מיועד להרצה חוזרת של הניסויים המרכזיים שמופיעים בדו"ח. אם קובצי הביניים כבר קיימים, אין צורך לבצע מחדש את שלבי הורדת הנתונים וחיתוך האודיו.

## 1. קבצים נדרשים

לניסויי Machine Learning נדרשים קובצי הפיצ'רים:

```text
master_extracted_features_archive_only.csv
master_extracted_features_free_text.csv
master_extracted_features_mixed.csv
```

לניסויי CNN נדרשות תיקיות הספקטרוגרמות:

```text
spectogram_logmel_archive_only/
spectogram_logmel_free_text/
spectogram_logmel_mixed/
```

ל-Multimodal CNN נדרשים גם הספקטרוגרמות וגם קובץ הפיצ'רים המתאים לאותו מצב.

## 2. שחזור ניסויי למידת מכונה

ההרצה מתבצעת דרך:

```text
main_for_machine_learning.py
```

יש לבחור את מצב הנתונים:

```python
RUN_MODE = "free_text"
```

בחירת הפיצ'רים נשלטת באמצעות:

```python
ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 30
```

ומצב הכוונון באמצעות:

```python
TUNING_MODE = "random_search"
```

יש להשתמש בערכים המתאימים לניסוי המתואר בדו"ח.

מחברת Kaggle:

```text
[להוסיף קישור]
```

## 3. שחזור המודל האקוסטי

יש לוודא שקיימת תיקיית הספקטרוגרמות המתאימה למצב ההרצה.

```text
spectogram_logmel_<mode>/
```

לאחר מכן מריצים:

```bash
python CNN.py
```

מחברת Kaggle:

```text
[להוסיף קישור]
```

## 4. שחזור המודל המולטי-מודאלי

יש לוודא שקיימים גם הספקטרוגרמות וגם קובץ הפיצ'רים של אותו מצב.

מריצים:

```bash
python Multimodal_CNN.py
```

מחברת Kaggle:

```text
[להוסיף קישור]
```

## 5. יצירת קובצי הקלט מחדש

אם רוצים להתחיל מקובצי האודיו:

1. מאחדים את מאגרי ה-Free Text בעזרת `group_sources_to_one_by_accent.py`.
2. חותכים את Speech Accent Archive בעזרת הקבצים שבתיקיית `cut_and_normalize/`.
3. יוצרים metadata בעזרת `metadata_generator.py`.
4. מנרמלים את האודיו בעזרת `normalize_all_for_2_folders.py`.
5. יוצרים את קובצי הפיצ'רים בעזרת `create_features.py`.
6. יוצרים את הספקטרוגרמות בעזרת `spectogram_creation.py`.

## 6. גרפים ותוצאות

חלק מהגרפים נוצרים כחלק מהרצת הניסוי.

ב-Kaggle הגרפים מוצגים בפלט של תאי המחברת. בהרצה מקומית דרך PyCharm הם נשמרים גם בתיקייה:

```text
graphs/
```

לכן בשחזור דרך Kaggle יש לבדוק את פלט התאים, ובהרצה מקומית יש לבדוק גם את תיקיית הגרפים.

תוצאות נוספות נשמרות תחת:

```text
results/
```

## 7. בדיקות לפני השוואה לדו"ח

לפני ההרצה יש לוודא שמשתמשים באותו מצב נתונים ובאותן הגדרות של הניסוי המקורי, ובפרט:

- מצב הנתונים.
- הפעלת או ביטול Feature Selection.
- מספר הפיצ'רים כאשר הבחירה מופעלת.
- מצב כוונון ההיפר-פרמטרים.
- קובץ הפיצ'רים או תיקיית הספקטרוגרמות המתאימים.
- ה-Dataset המתאים במחברת Kaggle.

במודלים שבהם קיימת אקראיות ייתכנו הבדלים קטנים בין הרצות.
