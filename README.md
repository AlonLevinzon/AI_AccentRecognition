# AI Accent Recognition

This project classifies English speech accents into four groups: **Chinese, Indian, Nigerian, and American (USA)**.

The project includes classical Machine Learning classifiers based on extracted audio features, an acoustic CNN based on Log-Mel spectrograms, and a Multimodal CNN that combines spectrograms with phonetic features.

## Project Structure

```text
AI_AccentRecognition/
|
|-- accents_audio/                              # Audio datasets organized by source
|   `-- accent_archive/                         # Speech Accent Archive
|       |-- chinese/                            # Audio files from Chinese-accented speakers
|       |-- hindi/                              # Audio files from Indian-accented speakers
|       |-- nigerian/                           # Audio files from Nigerian-accented speakers
|       `-- usa/                                # Audio files from American-accented speakers
|
|-- all_accents_data/                           # Original free-text datasets from different sources
|
|-- audio_by_accent_raw/                        # Free-text audio after merging sources by accent
|   |-- chinese/
|   |-- indian/
|   |-- nigerian/
|   `-- usa/
|
|-- archive_cnn_dataset/                        # dataset for uploading into archive_cnn notebook in kaggle
|-- audio_by_accent_raw_normalized/             # Normalized version of the free-text audio
|-- sentences_samples/                          # Speech Accent Archive recordings split into six sentences
|-- sentences_samples_normalized/               # Normalized version of the split sentences
|
|-- classifiers/                                # Machine Learning classifier implementations
|   |-- GNB.py                                  # Gaussian Naive Bayes
|   |-- KELM.py                                 # Kernel Extreme Learning Machine
|   |-- logistic_regression_classifier.py       # Logistic Regression
|   |-- random_forest_classifier.py             # Random Forest
|   |-- svm_classifier.py                       # Support Vector Machine
|   `-- XGBoost.py                              # XGBoost
|
|-- cut_and_normalize/                          # Audio splitting and normalization
|   |-- audio_time_split.py                     # Handles audio split timestamps
|   |-- cut_archive_to_sentences.py             # Splits Accent Archive recordings into sentences
|   |-- normalize_all_data_for_2_folders.py     # Normalizes the two main audio folders
|   `-- sentences_extractor.py                  # Locates/extracts sentences using transcription
|
|-- features/                                   # Feature extraction implementations
|   |-- audio_features.py                       # Acoustic features such as MFCC, ZCR, and formants
|   |-- Dynamic_features.py                     # Dynamic features such as Jitter and Shimmer
|   |-- extract_features_to_files.py            # Combines extracted features into each sample record
|   `-- phonetic_features.py                    # Creates phonetic/IPA-based features
|
|-- export_data/                                # Scripts for importing data from Hugging Face and other sources
|
|-- extra/                                      # Utility files, QA checks, and report-related material
|   |-- QA_FILES/                               # Validation checks, including Train/Validation/Test checks
|   |-- change_csv_archive_files.py             # Utility for modifying Archive CSV files
|   |-- visualize_features.py                   # Feature visualization utility
|   `-- voice_analysis_report.png               # Analysis output used during the project/report
|
|-- documentation/                              # Project documentation
|   |-- EXPERIMENT_REPRODUCTION_GUIDE.pdf
|   |-- PROGRAMMER_DOCUMENTATION.pdf
|   |-- USER_GUIDE.pdf
|   |-- appendix.pdf                            # Explanations of additional graphs generated during the experiments.
|
|-- machine_learning_official_archive_only/     # Metadata and inputs for archive_only mode
|-- machine_learning_official_free_text_only/   # Metadata and inputs for free_text mode
|-- machine_learning_official_mixed/            # Metadata and inputs for mixed mode
|
|-- spectrograms_logmel_archive_only/           # Log-Mel spectrograms for archive_only mode
|-- spectrograms_logmel_free_text/              # Log-Mel spectrograms for free_text mode
|-- spectrograms_logmel_mixed/                  # Log-Mel spectrograms for mixed mode
|
|-- graphs/                                     # Graphs saved during local runs
|-- results/                                    # Experiment results
|-- for final report/                           # Files used while preparing the final report
|
|-- archive_cnn.py                              # Compares Linear and Log-Mel spectrograms on archive dataset 
|-- metadata_generator.py                       # Creates metadata for the three dataset modes
|-- group_sources_to_one_by_accent.py           # Merges free-text sources by accent
|-- extract_features.py                         # Runs feature extraction and creates the master feature files
|-- create_spectogram.py                        # Creates Log-Mel spectrograms and saves them as .pt files
|-- K_fold.py                                   # Speaker-based Group K-Fold splitting and leakage prevention
|-- main_machine_learning.py                    # Runs the Machine Learning experiments
|-- main_cnn.py                                 # Runs the acoustic CNN
|-- main_multimodal_cnn.py                      # Runs the Multimodal CNN
|
|-- master_extracted_features_archive_only.csv  # Extracted features for archive_only mode
|-- master_extracted_features_free_text.csv     # Extracted features for free_text mode
|-- master_extracted_features_mixed.csv         # Extracted features for mixed mode
|
|-- README.md                                   # Project overview and file structure
|-- libs.txt                                    # Required Python packages
`-- EXPERIMENT_REPRODUCTION_GUIDE.pdf          # Guide for reproducing the experiments
```

Development-environment files such as `.idea` and `__pycache__`, as well as the model-weight files `best_acoustic_cnn_fold_1.pth` through `best_acoustic_cnn_fold_5.pth`, are intentionally omitted from the tree because they are not required for understanding the source-code structure.

## 📁 Datasets & Accent Sources

Below are the datasets used to train and evaluate our models for accent classification in this project, categorized by target accent :

### 1. Indian Accent (`indian`)
- **Primary Source (`indian_fleurs_accent`):**
  - Google FLEURS  [Indian English Voice (Hugging Face)](https://huggingface.co/datasets/Gbssreejith/indian-english-voice)
- **Secondary Source (`indian_mozzila_accent`):**
  - Mozilla Common Voice (Indian accent subset) / Common Voice Accent Subsets: [CommonVoice Indian Accent Dataset (Hugging Face)](https://huggingface.co/datasets/ishands/commonvoice-indian_accent) | [Mozilla Common Voice Corpus](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk)

The first dataset is identified as a subset of Google's FLEURS (English-India) based on its matching transcript content, recording metadata, and unique phrase patterns

---

### 2. Chinese Accent (`chinese`)
- **Primary Source (`chinese_speechOcean_accent`):**
  - OpenSLR SLR101 — Chinese Speech Corpus (speechOcean): [OpenSLR SLR101](https://www.openslr.org/101/)
- **Secondary Source (`chinese_ascend_accent`):**
  - CAiRE ASCEND Dataset (English transcript-only subset): [CAiRE ASCEND (Hugging Face)](https://huggingface.co/datasets/CAiRE/ASCEND)

---

### 3. Nigerian Accent (`nigerian`)
- **Primary Source (`nigerian_mozzila_accent`):**
  - Mozilla Common Voice / Nigerian Accented English Dataset: [Nigerian Accented English Dataset (Hugging Face)](https://huggingface.co/datasets/benjaminogbonna/nigerian_accented_english_dataset) | [Mozilla Common Voice](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk)
- **Secondary Source (`nigerian_slr_70_accent`):**
  - OpenSLR SLR70 — High-quality African English Speech: [OpenSLR SLR70](https://www.openslr.org/70/)

---

### 4. US Accent (`usa`)
- **Primary Source (`usa_libriSpeech_accent`):**
  - OpenSLR SLR12 — American English Accents / LibriSpeech: [OpenSLR SLR12 (American English Accents)](https://www.openslr.org/12)
- **Secondary Source (`usa_mozzila_accent`):**
  - Mozilla Common Voice (US accent subset):[CommonVoice US Accent Dataset (Hugging Face)](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)| [Mozilla Common Voice Corpus](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk)

---

### 🌐 General & Multi-Accent Archives
- Speech Accent Archive: [GMU Speech Accent Archive](http://accent.gmu.edu) | [Kaggle Speech Accent Archive](https://www.kaggle.com/datasets/rtatman/speech-accent-archive)


## Workflow

The data comes from two main types of sources. Speech Accent Archive recordings are stored under `accents_audio/accent_archive/`, while the original free-text datasets are stored under `all_accents_data/`.

`group_sources_to_one_by_accent.py` combines the free-text sources according to the selected dataset composition and organizes the resulting audio files by accent under `audio_by_accent_raw/`.

Speech Accent Archive recordings are split into six sentences using the scripts under `cut_and_normalize/cut_archive_to_sentences`. The resulting samples are stored under `sentences_samples/`. `metadata_generator.py` then creates the metadata required for the three experiment modes: `archive_only`, `free_text`, and `mixed`.

The script cut_and_normalize/normalize_all_data_for_2_folders.py performs unified preprocessing and normalization on both the sentences_samples and audio_by_accent_raw directories. It generates standardized output directories appended with the _normalized suffix.

Before the experiment paths diverge, `extract_features.py` runs the feature-extraction code implemented under `features/` and creates `master_extracted_features_*.csv` file for each mode.

For the **Machine Learning** path, `main_machine_learning.py` runs the classifiers implemented under `classifiers/`.

For the **Deep Learning** path, `create_spectogram.py` creates the Log-Mel spectrogram files. The acoustic model is then run with `main_cnn.py`, while the multimodal model is run with `main_multimodal_cnn.py`.

##  Runtime Environment (Kaggle Notebooks or PyCharm)

It is highly recommended to run the four main experiment notebooks on Kaggle. Before running a notebook, make sure that its corresponding dataset is attached to the notebook.


[Machine Learning notebook]( https://www.kaggle.com/code/alonlev001/main-for-machine-learning)

[CNN notebook](https://www.kaggle.com/code/alonlev001/ai-accent-classifier-cnn)

[Multimodal CNN notebook](https://www.kaggle.com/code/alonlev001/ai-accent-classifier-multimodal-cnn)

[archive CNN notebook (for spectrogram compare)](https://www.kaggle.com/code/alonlev001/archive-cnn)

**For users running this project via PyCharm**: Please review the libs.txt file prior to setup. Note that the package installation process and library compatibility may vary depending on your PyCharm version, Python interpreter configuration, and operating system.

For additional information please review the files in the documentation folder.

## Results and Graphs

When the experiments are run on Kaggle, some graphs are displayed directly in the notebook output. When the project is run locally in PyCharm, the generated graphs are also saved under `graphs/`.

Additional experiment outputs are stored under `results/`.

The `appendix.pdf` document, located in the `documentation` folder, provides explanations of additional graphs generated during the experiments that were not included in the main project report. These graphs are available in the saved experiment outputs.

## Submission Documentation

`README.md`, `libs.txt`, and `EXPERIMENT_REPRODUCTION_GUIDE.docx` are located in the project root directory.

The user guide and programmer documentation are located under `documentation/`.
