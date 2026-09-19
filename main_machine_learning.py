import os
import sys
import warnings
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# התעלמות מאזהרות
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*max_iter.*')

# הגדרת נתיבי עבודה מקומיים ל-PyCharm
WORKING_DIR = Path.cwd()
PROJECT_DATA_DIR = Path.cwd()

if str(PROJECT_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DATA_DIR))

# ייבוא מודולים פנימיים
from classifiers.GNB import get_gaussian_nb_classifier
from classifiers.logistic_regression_classifier import get_logistic_regression_classifier
from classifiers.random_forest_classifier import get_random_forest_classifier
from classifiers.svm_classifier import get_svm_classifier
from classifiers.XGBoost import get_xgboost_classifier

import K_fold
from K_fold import (
    build_free_text_only_metadata,
    build_mixed_metadata,
    run_nested_group_kfold_generator,
    load_exact_metadata,
    clean_speaker_id,
    sanitize_accent_label,
    clean_filename_path,
    clean_text_quotes,
    sample_dataset_safely
)
from features.extract_features_to_files import import_vector_to_file

# הגדרת נתיבי הפלט עבור K_fold
K_fold.FREE_TEXT_OUT_DIR = WORKING_DIR / "machine_learning_official_free_text_only"
K_fold.MIXED_OUT_DIR = WORKING_DIR / "machine_learning_official_mixed"
K_fold.ARCHIVE_OUT_DIR = WORKING_DIR / "machine_learning_official_archive_only"

# =====================================================================
# CONFIGURATION
# =====================================================================
RUN_MODE = "free_text"  # "free_text", "mixed", or "archive_only"
ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 30
TUNING_MODE = "random_search"  # "random_search", "custom", or "default"

CUSTOM_HYPERPARAMETERS = {
    'Logistic Regression': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
    'SVM (RBF)': {'C': 10, 'gamma': 'scale'},
    'Random Forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1,
                      'max_features': 'sqrt'},
    'XGBoost': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'Kernel ELM': {'C': 10, 'gamma': 0.01}
}

extracted_filename = f"master_extracted_features_{RUN_MODE}.csv"
EXTRACTED_FEATURES_MASTER_CSV = WORKING_DIR / extracted_filename


# =====================================================================
# HELPER CLASSES & PIPELINE FUNCTIONS
# =====================================================================
class KernelELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma=0.1, kernel='rbf'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        distances_sq = np.maximum(distances_sq, 0.0)
        return np.exp(-self.gamma * distances_sq)

    def fit(self, X, y):
        self.X_train_ = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        y_arr = y_arr.flatten()

        self.classes_ = np.unique(y_arr)
        num_samples = self.X_train_.shape[0]
        num_classes = len(self.classes_)

        y_label_matrix = np.zeros((num_samples, num_classes), dtype=np.float64)
        for idx, cls_val in enumerate(self.classes_):
            y_label_matrix[y_arr == cls_val, idx] = 1.0

        K = self.rbf_kernel(self.X_train_, self.X_train_)
        reg_matrix = K + (1.0 / self.C) * np.eye(num_samples) + 1e-7 * np.eye(num_samples)

        try:
            self.beta_ = np.linalg.solve(reg_matrix, y_label_matrix)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.pinv(reg_matrix) @ y_label_matrix

        return self

    def predict(self, X):
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        class_indices = np.argmax(raw_output, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        exp_scores = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def build_archive_only_metadata():
    print("\n--- Building Archive Only Massive Metadata ---")
    all_records = []
    accents = ['chinese', 'indian', 'nigerian', 'usa']
    sentences_dir = PROJECT_DATA_DIR / "sentences_samples_normalized"
    archive_out_dir = WORKING_DIR / "machine_learning_official_archive_only"

    stella_sentences = [
        "Please call Stella Ask her to bring these things",
        "with her from the store Six spoons of fresh snow peas",
        "five thick slabs of blue cheese and maybe a snack for her brother Bob",
        "We also need a small plastic snake and a big toy frog for the kids",
        "She can scoop these things into three red bags",
        "and we will go meet her Wednesday at the train station"
    ]

    for accent in accents:
        accent_dir = sentences_dir / accent
        if not accent_dir.exists():
            continue

        df_raw = load_exact_metadata(accent_dir, accent)
        if df_raw is None:
            continue

        speaker_col_name = next((k for k in ['speakerid', 'speaker_id', 'client_id', 'id'] if k in df_raw.columns),
                                None)
        df_unique_speakers = df_raw.copy()
        if speaker_col_name:
            df_unique_speakers['temp_spk'] = df_unique_speakers[speaker_col_name].apply(clean_speaker_id)
            df_unique_speakers = df_unique_speakers.drop_duplicates(subset=['temp_spk'])

        sentence_records = []
        for idx, row in df_unique_speakers.iterrows():
            raw_filename = next(
                (df_unique_speakers.at[idx, k] for k in ['filename', 'file_name', 'audio', 'file', 'wav_path'] if
                 k in df_unique_speakers.columns and pd.notna(df_unique_speakers.at[idx, k])), "")
            if not raw_filename:
                continue

            pure_filename = clean_filename_path(raw_filename)
            speaker_raw = next((df_unique_speakers.at[idx, k] for k in ['speakerid', 'speaker_id', 'client_id', 'id'] if
                                k in df_unique_speakers.columns and pd.notna(df_unique_speakers.at[idx, k])), "")
            speaker_id = clean_speaker_id(speaker_raw)
            accent_raw = df_unique_speakers.at[idx, 'accent'] if 'accent' in df_unique_speakers.columns else ""
            accent_label = sanitize_accent_label(accent_raw, accent)
            base_name, _ = os.path.splitext(pure_filename)

            for i in range(6):
                archive_filename = f"{base_name}_part_{i}.wav"
                path_with_folder = accent_dir / f"sentence_{i}" / archive_filename
                path_direct = accent_dir / archive_filename
                valid_path = path_with_folder if path_with_folder.exists() else (
                    path_direct if path_direct.exists() else None)

                if valid_path is not None:
                    final_text = clean_text_quotes(stella_sentences[i])
                    sentence_records.append({
                        'filename': archive_filename, 'file_path': str(valid_path),
                        'speaker_id': speaker_id, 'accent': accent_label,
                        'text': final_text, 'source_type': 'archive'
                    })

        df_sentence = pd.DataFrame(sentence_records)
        if df_sentence.empty:
            continue

        df_sentence_capped = sample_dataset_safely(df_sentence, target_limit=60, mode='speakers')
        all_records.append(df_sentence_capped)

    if all_records:
        df_final = pd.concat(all_records, ignore_index=True)
        os.makedirs(archive_out_dir, exist_ok=True)
        df_final[['filename', 'file_path', 'speaker_id', 'accent', 'text']].to_csv(
            archive_out_dir / "massive_metadata_archive_only.csv", index=False)
        return df_final
    return pd.DataFrame()


def pre_extract_all_features_once(df_master):
    if EXTRACTED_FEATURES_MASTER_CSV.exists():
        print(f"──► [CACHED DATA DETECTED]: Loading pre-extracted features from {EXTRACTED_FEATURES_MASTER_CSV}")
        return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)

    print(f"\n⚡⚡⚡ [FIRST RUN INITIALIZATION]: Extracting acoustic features for ALL {len(df_master)} files once...")
    temp_master_meta = WORKING_DIR / "temp_master_metadata_pipeline.csv"
    df_master.to_csv(temp_master_meta, index=False, encoding='utf-8-sig')

    audio_base_dir = PROJECT_DATA_DIR / "audio_by_accent_raw_normalized"
    import_vector_to_file(str(temp_master_meta), str(audio_base_dir))

    default_output = WORKING_DIR / "accent_features_dataset.csv"
    if default_output.exists():
        if EXTRACTED_FEATURES_MASTER_CSV.exists():
            os.remove(EXTRACTED_FEATURES_MASTER_CSV)
        os.rename(default_output, EXTRACTED_FEATURES_MASTER_CSV)
        if temp_master_meta.exists():
            os.remove(temp_master_meta)
        print(f"──► [SUCCESS]: Cached unified dataset saved successfully at {EXTRACTED_FEATURES_MASTER_CSV}")
    else:
        raise FileNotFoundError("Feature extraction subroutine failed.")

    return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)


def filter_features_by_metadata_split(df_master_features, df_split_meta):
    df_merged = pd.merge(
        df_split_meta[['filename', 'file_path', 'speaker_id', 'text', 'accent']],
        df_master_features.drop(columns=['speaker_id', 'text', 'accent'], errors='ignore'),
        on='filename', how='inner'
    )
    if len(df_merged) < len(df_split_meta):
        missing_count = len(df_split_meta) - len(df_merged)
        print(f"⚠️ [WARNING]: נשמטו {missing_count} שורות ב-Fold הנוכחי!")
    return df_merged


def apply_anova_feature_selection(df_train, available_features, top_k=30):
    X_tr = df_train[available_features].fillna(0)
    non_constant_features = [col for col in available_features if X_tr[col].nunique() > 1]
    if not non_constant_features:
        return available_features[:top_k]

    X_filtered = X_tr[non_constant_features]
    y_tr = df_train['accent']

    f_scores, _ = f_classif(X_filtered, y_tr)
    anova_df = pd.DataFrame({'Feature': non_constant_features, 'F_Score': f_scores})
    anova_df['F_Score'] = anova_df['F_Score'].fillna(0)
    return anova_df.sort_values(by='F_Score', ascending=False)['Feature'].head(top_k).tolist()


def load_and_scale_fold_matrices(df_master_features, df_train_meta, df_val_meta, df_test_meta):
    df_train = filter_features_by_metadata_split(df_master_features, df_train_meta)
    df_val = filter_features_by_metadata_split(df_master_features, df_val_meta)
    df_test = filter_features_by_metadata_split(df_master_features, df_test_meta)

    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio',
                    'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    raw_feature_candidates = [col for col in df_train.columns if col not in exclude_cols]

    if ENABLE_FEATURE_SELECTION and len(raw_feature_candidates) > TOP_K_FEATURES:
        selected_features = apply_anova_feature_selection(df_train, raw_feature_candidates, top_k=TOP_K_FEATURES)
    else:
        selected_features = raw_feature_candidates

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[selected_features]).astype(float)
    X_val = scaler.transform(df_val[selected_features]).astype(float)
    X_test = scaler.transform(df_test[selected_features]).astype(float)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['accent'])
    y_val = le.transform(df_val['accent'])
    y_test = le.transform(df_test['accent'])

    return X_train, y_train, X_val, y_val, X_test, y_test, le, selected_features


def tune_model_hyperparameters(model_name, X_train, y_train, fold_tuning_history, n_iter=20, cv_folds=3):
    if TUNING_MODE == "default":
        return None, {}

    if TUNING_MODE == "custom":
        custom_params = CUSTOM_HYPERPARAMETERS.get(model_name, {})
        if model_name == 'Logistic Regression':
            return LogisticRegression(max_iter=3000, random_state=42, **custom_params), custom_params
        elif model_name == 'SVM (RBF)':
            return SVC(probability=True, random_state=42, **custom_params), custom_params
        elif model_name == 'Random Forest':
            return RandomForestClassifier(random_state=42, n_jobs=-1, **custom_params), custom_params
        elif model_name == 'XGBoost':
            return XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1, **custom_params), custom_params
        elif model_name == 'Kernel ELM':
            return KernelELMClassifier(**custom_params), custom_params
        elif model_name == 'Gaussian NB':
            return GaussianNB(**custom_params), custom_params
        return None, {}

    if TUNING_MODE == "random_search":
        if model_name == 'Logistic Regression':
            base_model = LogisticRegression(max_iter=3000, random_state=42)
            param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg']}
        elif model_name == 'SVM (RBF)':
            base_model = SVC(probability=True, random_state=42)
            param_grid = {'C': [0.1, 1, 10, 50, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]}
        elif model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30],
                          'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                          'max_features': ['sqrt', 'log2']}
        elif model_name == 'XGBoost':
            base_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 9],
                          'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.7, 0.8, 1.0],
                          'colsample_bytree': [0.7, 0.8, 1.0]}
        elif model_name == 'Kernel ELM':
            base_model = KernelELMClassifier()
            param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0]}
        elif model_name == 'Gaussian NB':
            base_model = GaussianNB()
            param_grid = {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
                          'priors': [None, (0.25, 0.25, 0.25, 0.25), (0.1, 0.1, 0.1, 0.7), (0.4, 0.2, 0.2, 0.2)]}
        else:
            return None, {}

        search = RandomizedSearchCV(
            estimator=base_model, param_distributions=param_grid,
            n_iter=n_iter, cv=cv_folds, scoring='accuracy', random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)

        for params, mean_score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score']):
            param_key = tuple(sorted(params.items()))
            fold_tuning_history[model_name][param_key].append(mean_score)

        return search.best_estimator_, search.best_params_

    return None, {}


def export_and_print_best_hyperparameters(fold_tuning_history):
    if TUNING_MODE != "random_search" or not fold_tuning_history:
        return

    print("\n" + "=" * 80)
    print("🏆 OPTIMAL HYPERPARAMETERS REPORT (AVERAGED ACROSS ALL 10 FOLDS)")
    print("=" * 80)

    report_rows = []
    for model_name, history in fold_tuning_history.items():
        if not history:
            continue

        best_param_dict = None
        best_avg_accuracy = -1.0

        for param_tuple, scores in history.items():
            avg_score = np.mean(scores) * 100
            if avg_score > best_avg_accuracy:
                best_avg_accuracy = avg_score
                best_param_dict = dict(param_tuple)

        print(f"\n📌 Classifier: [{model_name}]")
        print(f"   • Mean Cross-Validation Accuracy: {best_avg_accuracy:.2f}%")
        print(f"   • Best Hyperparameters:")
        for k, v in best_param_dict.items():
            print(f"       - {k}: {v}")

        report_rows.append({
            'Classifier': model_name,
            'Best_Averaged_Accuracy_%': round(best_avg_accuracy, 2),
            'Optimal_Hyperparameters': str(best_param_dict)
        })

    df_best_report = pd.DataFrame(report_rows)
    report_csv_path = WORKING_DIR / "best_hyperparameters_summary.csv"
    df_best_report.to_csv(report_csv_path, index=False)
    print(f"\n💾 Dedicated Hyperparameters Report exported successfully -> {report_csv_path}")


def generate_hyperparameter_averaged_plots(fold_tuning_history):
    if TUNING_MODE != "random_search":
        return

    for model_name, history in fold_tuning_history.items():
        if not history:
            continue

        averaged_data = []
        for param_tuple, scores in history.items():
            param_dict = dict(param_tuple)
            param_dict['mean_accuracy'] = np.mean(scores) * 100
            averaged_data.append(param_dict)

        df_params = pd.DataFrame(averaged_data)
        plt.figure(figsize=(9, 5), dpi=300)

        if model_name == 'Gaussian NB':
            df_params['var_smoothing'] = pd.to_numeric(df_params['var_smoothing'], errors='coerce')
            subset = df_params.sort_values(by='var_smoothing')
            plt.plot(subset['var_smoothing'], subset['mean_accuracy'], marker='o', color='#377eb8',
                     label="var_smoothing")
            plt.xscale('log')
            plt.xlabel("Variance Smoothing Parameter (Log Scale - Regularization Power)", fontsize=11)
            plt.ylabel("10-Fold Mean Cross-Validation Accuracy (%)", fontsize=11)

        elif model_name == 'Kernel ELM':
            df_params['C'] = pd.to_numeric(df_params['C'], errors='coerce')
            for gamma_val in df_params['gamma'].unique():
                subset = df_params[df_params['gamma'] == gamma_val].sort_values(by='C')
                plt.plot(subset['C'], subset['mean_accuracy'], marker='o', label=f"RBF Kernel Gamma={gamma_val}")
            plt.xscale('log')
            plt.xlabel("Regularization Cost Parameter C (Log Scale)", fontsize=11)
            plt.ylabel("10-Fold Mean Cross-Validation Accuracy (%)", fontsize=11)

        elif model_name == 'Logistic Regression':
            df_params['C'] = pd.to_numeric(df_params['C'], errors='coerce')
            for solver in df_params['solver'].unique():
                subset = df_params[df_params['solver'] == solver].sort_values(by='C')
                plt.plot(subset['C'], subset['mean_accuracy'], marker='o', label=f"Optimization Solver={solver}")
            plt.xscale('log')
            plt.xlabel("Inverse Regularization Strength C (Log Scale)", fontsize=11)
            plt.ylabel("10-Fold Mean Cross-Validation Accuracy (%)", fontsize=11)

        elif model_name == 'SVM (RBF)':
            df_params['C'] = pd.to_numeric(df_params['C'], errors='coerce')
            for gamma_val in df_params['gamma'].unique():
                subset = df_params[df_params['gamma'] == gamma_val].sort_values(by='C')
                plt.plot(subset['C'], subset['mean_accuracy'], marker='o', label=f"Kernel Gamma={gamma_val}")
            plt.xscale('log')
            plt.xlabel("Soft Margin Cost Parameter C (Log Scale)", fontsize=11)
            plt.ylabel("10-Fold Mean Cross-Validation Accuracy (%)", fontsize=11)

        elif model_name == 'Random Forest':
            df_params['n_estimators'] = pd.to_numeric(df_params['n_estimators'], errors='coerce')
            df_params['max_depth_str'] = df_params['max_depth'].apply(
                lambda x: 'None' if pd.isna(x) or x is None else str(x))
            for depth in df_params['max_depth_str'].unique():
                subset = df_params[df_params['max_depth_str'] == depth].sort_values(by='n_estimators')
                plt.plot(subset['n_estimators'], subset['mean_accuracy'], marker='s', label=f"Max Tree Depth={depth}")
            plt.xlabel("Number of Decision Trees (n_estimators)", fontsize=11)
            plt.ylabel("10-Fold Mean Cross-Validation Accuracy (%)", fontsize=11)

        elif model_name == 'XGBoost':
            df_params['learning_rate'] = pd.to_numeric(df_params['learning_rate'], errors='coerce')
            for depth in df_params['max_depth'].unique():
                subset = df_params[df_params['max_depth'] == depth].sort_values(by='learning_rate')
                plt.plot(subset['learning_rate'], subset['mean_accuracy'], marker='^', label=f"Max Tree Depth={depth}")
            plt.xlabel("Gradient Boosting Learning Rate (Eta)", fontsize=11)
            plt.ylabel("10-Fold Mean Cross-Validation Accuracy (%)", fontsize=11)

        plt.title(f"Hyperparameter Optimization Profile - {model_name}", fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)

        chart_path = WORKING_DIR / f"hyperparameter_impact_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


def generate_overall_plots(df_summary, df_feat_summary):
    # 1. Classifier Overall Performance Plot
    plt.figure(figsize=(10, 5), dpi=300)
    models = df_summary['Classifier']
    acc_col = 'Overall Accuracy' if 'Overall Accuracy' in df_summary.columns else 'Mean Test Accuracy (%)'
    accuracies = [float(str(val).split('%')[0]) for val in df_summary[acc_col]]

    plt.bar(models, accuracies, color='#4c72b0', edgecolor='black', alpha=0.85)
    plt.title("Overall Classifier Test Accuracy Comparison (Across 10 Nested Group Folds)", fontsize=13,
              fontweight='bold')
    plt.xlabel("Machine Learning Classifier Model", fontsize=11, labelpad=10)
    plt.ylabel("Mean Test Accuracy Percentage (%)", fontsize=11)
    plt.ylim(0, 100)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 1.5, f"{v:.2f}%", ha='center', fontweight='bold', fontsize=9)

    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    chart_path = WORKING_DIR / "classifier_accuracy_comparison.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # 2. Top Selected Features Frequency Plot
    plt.figure(figsize=(10, 6), dpi=300)
    feat_col = df_feat_summary.columns[0]
    freq_col = df_feat_summary.columns[1]
    top_feats = df_feat_summary.head(15)

    plt.barh(top_feats[feat_col], top_feats[freq_col], color='#2ca02c', edgecolor='black', alpha=0.85)
    plt.xlabel("Selection Frequency (Selected Out of 10 Cross-Validation Folds)", fontsize=11)
    plt.ylabel("Acoustic Speech Feature Name", fontsize=11)
    plt.title("Top 15 Most Important Acoustic Features Selected by ANOVA", fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    feat_chart_path = WORKING_DIR / "top_features_frequency.png"
    plt.savefig(feat_chart_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    print("====================================================")
    print(f"HIGH-SPEED Multi-Classifier Nested Group K-Fold Loop ({RUN_MODE})")
    print("====================================================")

    metadata_file_name = f"massive_metadata_{RUN_MODE}.csv"
    possible_paths = [
        PROJECT_DATA_DIR / f"machine_learning_official_{RUN_MODE}_only" / metadata_file_name,
        PROJECT_DATA_DIR / f"machine_learning_official_{RUN_MODE}" / metadata_file_name,
        PROJECT_DATA_DIR / metadata_file_name
    ]

    df_master = None
    for meta_path in possible_paths:
        if meta_path.exists():
            print(f"📂 [LOAD SUCCESS]: Found existing metadata directly at: {meta_path}")
            df_master = pd.read_csv(meta_path)
            break

    if df_master is None:
        print("⚠️ Metadata CSV not found, triggering builder function...")
        if RUN_MODE == "free_text":
            df_master = build_free_text_only_metadata()
        elif RUN_MODE == "mixed":
            df_master = build_mixed_metadata()
        elif RUN_MODE == "archive_only":
            df_master = build_archive_only_metadata()

    if df_master is None or df_master.empty:
        sys.exit("❌ Error: Master metadata dataframe is empty.")

    df_master_features = pre_extract_all_features_once(df_master)

    # 🛑 בדיקת זליגת דוברים
    print("\n🔍 מריץ בדיקת זליגת דוברים קפדנית...")
    temp_fold_gen = run_nested_group_kfold_generator(df_master)
    for fold_i, train_m, val_m, test_m in temp_fold_gen:
        train_spks, val_spks, test_spks = set(train_m['speaker_id'].unique()), set(val_m['speaker_id'].unique()), set(
            test_m['speaker_id'].unique())
        if train_spks.intersection(val_spks) or train_spks.intersection(test_spks) or val_spks.intersection(test_spks):
            sys.exit(f"❌ [PROCESS HALTED]: Speaker leakage detected in Fold {fold_i}!")

    print("✅ [VERIFICATION PASSED]: אפס זליגת דוברים!\n")

    model_names = ['Gaussian NB', 'Logistic Regression', 'SVM (RBF)', 'Random Forest', 'XGBoost', 'Kernel ELM']
    accents = ['chinese', 'indian', 'nigerian', 'usa']

    performance_tracker = {
        m: {'overall': [], 'train_acc': [], 'per_accent': {a: [] for a in accents}, 'y_true_all': [], 'y_pred_all': []}
        for m in model_names
    }

    fold_tuning_history = defaultdict(lambda: defaultdict(list))
    all_selected_features_history = []

    fold_generator = run_nested_group_kfold_generator(df_master)

    for fold_idx, df_train_meta, df_val_meta, df_test_meta in fold_generator:
        print(f"\n######################################################################")
        print(f" PROCESSING NESTED CROSS-VALIDATION FOLD SYSTEM: [FOLD {fold_idx}/10]")
        print(f"######################################################################")

        try:
            X_train, y_train, X_val, y_val, X_test, y_test, le, selected_feats = load_and_scale_fold_matrices(
                df_master_features, df_train_meta, df_val_meta, df_test_meta
            )
            all_selected_features_history.extend(selected_feats)

            models = {
                'Gaussian NB': get_gaussian_nb_classifier(),
                'Logistic Regression': get_logistic_regression_classifier(),
                'SVM (RBF)': get_svm_classifier(),
                'Random Forest': get_random_forest_classifier(),
                'XGBoost': get_xgboost_classifier(),
                'Kernel ELM': KernelELMClassifier()
            }

            for model_name, model_obj in models.items():
                tuned_estimator, best_params = tune_model_hyperparameters(
                    model_name, X_train, y_train, fold_tuning_history, n_iter=20
                )
                if tuned_estimator is not None:
                    model_obj = tuned_estimator

                model_obj.fit(X_train, y_train)

                y_train_pred = model_obj.predict(X_train)
                train_acc = accuracy_score(y_train, y_train_pred)

                y_pred = model_obj.predict(X_test)
                overall_acc = accuracy_score(y_test, y_pred)

                performance_tracker[model_name]['train_acc'].append(train_acc)
                performance_tracker[model_name]['overall'].append(overall_acc)
                performance_tracker[model_name]['y_true_all'].extend(y_test)
                performance_tracker[model_name]['y_pred_all'].extend(y_pred)

                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True,
                                               zero_division=0)
                for acc_label in accents:
                    if acc_label in report:
                        performance_tracker[model_name]['per_accent'][acc_label].append(report[acc_label]['recall'])

                print(
                    f"    [{model_name:<19}] Train Acc: {train_acc * 100:.2f}% | Test Acc: {overall_acc * 100:.2f}% | Gap: {(train_acc - overall_acc) * 100:.2f}%")

        except Exception as e:
            print(f"Error executing processing loop inside Fold {fold_idx}: {e}")
            traceback.print_exc()

    # --- SUMMARY TABLE ---
    summary_rows = []
    for model_name in model_names:
        tracker = performance_tracker[model_name]
        if not tracker['overall']:
            continue

        mean_train = np.mean(tracker['train_acc']) * 100
        mean_overall = np.mean(tracker['overall']) * 100
        std_overall = np.std(tracker['overall']) * 100

        row = {
            'Classifier': model_name,
            'Best/Mean Train Acc': f"{mean_train:.2f}%",
            'Overall Accuracy': f"{mean_overall:.2f}% ± {std_overall:.2f}%",
            'Generalization Gap (%)': f"{(mean_train - mean_overall):.2f}%"
        }
        for acc_label in accents:
            scores = tracker['per_accent'][acc_label]
            row[f"{acc_label.capitalize()} Accuracy"] = f"{np.mean(scores) * 100:.2f}%" if scores else "N/A"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\n", df_summary.to_string(index=False))

    # --- CONFUSION MATRICES ---
    for model_name in model_names:
        tracker = performance_tracker[model_name]
        if tracker['y_true_all']:
            cm = confusion_matrix(tracker['y_true_all'], tracker['y_pred_all'])
            print(f"\n--- {model_name} Confusion Matrix ---")
            print(f"Classes order: {le.classes_}\n{cm}")

    export_and_print_best_hyperparameters(fold_tuning_history)

    # --- FEATURE SELECTION SUMMARY ---
    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio',
                    'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    all_possible_features = [col for col in df_master_features.columns if col not in exclude_cols]
    feat_counts = Counter(all_selected_features_history)
    full_feat_counts = {feat: feat_counts.get(feat, 0) for feat in all_possible_features}

    df_feat_summary = pd.DataFrame(list(full_feat_counts.items()), columns=['Feature', 'Selection_Frequency_Out_Of_10'])
    df_feat_summary = df_feat_summary.sort_values(by=['Selection_Frequency_Out_Of_10', 'Feature'],
                                                  ascending=[False, True])

    out_csv_path = WORKING_DIR / "top_selected_features_summary.csv"
    df_feat_summary.to_csv(out_csv_path, index=False)

    # --- 2D PCA PROJECTION (TOP 30) ---
    top_30_features = df_feat_summary.head(30)['Feature'].tolist()
    if len(top_30_features) > 0:
        X_top30 = df_master_features[top_30_features].fillna(0)
        y_labels = df_master_features['accent'].astype(str).str.capitalize()

        scaler_pca = StandardScaler()
        X_top30_scaled = scaler_pca.fit_transform(X_top30)

        pca_30 = PCA(n_components=2, random_state=42)
        X_pca_30 = pca_30.fit_transform(X_top30_scaled)

        exp_var_1 = pca_30.explained_variance_ratio_[0] * 100
        exp_var_2 = pca_30.explained_variance_ratio_[1] * 100
        total_exp_var = np.sum(pca_30.explained_variance_ratio_) * 100

        plt.figure(figsize=(11, 8), dpi=300)
        sns.scatterplot(x=X_pca_30[:, 0], y=X_pca_30[:, 1], hue=y_labels, palette="Set1", alpha=0.65, s=50)
        plt.title(
            f"2D PCA Projection of Top 30 ANOVA Acoustic Features\nTotal Explained Variance: {total_exp_var:.2f}% (PC1: {exp_var_1:.2f}%, PC2: {exp_var_2:.2f}%)",
            fontsize=13, fontweight='bold', pad=15)
        plt.xlabel(f"Principal Component 1 ({exp_var_1:.2f}% Variance)", fontsize=11, fontweight='bold')
        plt.ylabel(f"Principal Component 2 ({exp_var_2:.2f}% Variance)", fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Accent", loc='best', frameon=True)
        plt.tight_layout()

        pca_top30_save_path = WORKING_DIR / "top_30_features_pca_2d.png"
        plt.savefig(pca_top30_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    generate_overall_plots(df_summary, df_feat_summary)
    generate_hyperparameter_averaged_plots(fold_tuning_history)


if __name__ == "__main__":
    main()