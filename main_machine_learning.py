import os
import sys
import shutil
import warnings
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

import librosa
import soundfile as sf

# Safe module checks for optional graph generation
try:
    import umap
except ImportError:
    umap = None

try:
    import shap
except ImportError:
    shap = None

# ==============================================================================
# CONFIGURATION & SYSTEM SETUP
# ==============================================================================

# 1. Dataset Subset Execution Mode: "free_text", "mixed", or "archive_only"
RUN_MODE = "free_text"

# 2. Hyperparameter Tuning Strategy Mode: "random_search", "custom", or "default"
TUNING_MODE = "random_search"

ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 30

CUSTOM_HYPERPARAMETERS = {
    'Logistic Regression': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
    'SVM (RBF)': {'C': 10, 'gamma': 'scale'},
    'Random Forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1,
                      'max_features': 'sqrt'},
    'XGBoost': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'Kernel ELM': {'C': 10, 'gamma': 0.01}
}

# 3. Local Project Directory Configuration
PROJECT_DIR = Path.cwd().resolve()
WORKING_DIR = PROJECT_DIR
PROJECT_DATA_DIR = PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 4. Setup Graphs Directory & Clear Previous Artifacts per Run Mode
GRAPHS_DIR = PROJECT_DIR / "graphs" / RUN_MODE
if GRAPHS_DIR.exists():
    shutil.rmtree(GRAPHS_DIR)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"⚙️ PIPELINE EXECUTION CONFIGURATION:")
print(f"   • RUN_MODE     : '{RUN_MODE}'")
print(f"   • TUNING_MODE  : '{TUNING_MODE}'")
print(f"   • PROJECT_ROOT : {PROJECT_DIR}")
print(f"   • GRAPHS_DIR   : {GRAPHS_DIR} (Cleared & Prepared)")
print("=" * 80)

# Suppress non-critical execution warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*max_iter.*')

# Import custom project modules
import K_fold
from K_fold import (
    build_free_text_only_metadata,
    build_mixed_metadata,
    build_archive_only_metadata,
    run_nested_group_kfold_generator,
    load_exact_metadata,
    clean_speaker_id,
    sanitize_accent_label,
    clean_filename_path,
    clean_text_quotes,
    sample_dataset_safely
)
from classifiers.GNB import get_gaussian_nb_classifier
from classifiers.logistic_regression_classifier import get_logistic_regression_classifier
from classifiers.random_forest_classifier import get_random_forest_classifier
from classifiers.svm_classifier import get_svm_classifier
from classifiers.XGBoost import get_xgboost_classifier
from features.extract_features_to_files import import_vector_to_file

# Set output directories within K_fold module
K_fold.FREE_TEXT_OUT_DIR = WORKING_DIR / "machine_learning_official_free_text_only"
K_fold.MIXED_OUT_DIR = WORKING_DIR / "machine_learning_official_mixed"
K_fold.ARCHIVE_OUT_DIR = WORKING_DIR / "machine_learning_official_archive_only"

# Locate Master Extracted Features CSV
extracted_filename = f"master_extracted_features_{RUN_MODE}.csv"
EXTRACTED_FEATURES_MASTER_CSV = PROJECT_DIR / extracted_filename

if EXTRACTED_FEATURES_MASTER_CSV.exists():
    print(f"🎯 [CACHED FEATURES FOUND]: {EXTRACTED_FEATURES_MASTER_CSV}")
else:
    print(f"⚠️ Target path for features: {EXTRACTED_FEATURES_MASTER_CSV}")


# ==============================================================================
# CUSTOM CLASSIFIER DEFINITIONS
# ==============================================================================

# This class implements a custom Kernel Extreme Learning Machine (KELM) classifier compatible with scikit-learn.
# It helps the project by providing fast training and high-dimensional kernel learning for accent classification.
class KernelELMClassifier(BaseEstimator, ClassifierMixin):
    """
    Implements a Kernel Extreme Learning Machine (KELM) classifier.
    """

    # Constructor function that initializes the regularization parameter (C), kernel scale (gamma), and kernel type.
    def __init__(self, C=1.0, gamma=0.1, kernel='rbf'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

    # Helper function that computes the Radial Basis Function (RBF) kernel matrix between sample feature vectors.
    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        distances_sq = np.maximum(distances_sq, 0.0)
        return np.exp(-self.gamma * distances_sq)

    # Main training function that calculates analytical output weights in one step using regularized ridge regression.
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

    # Prediction function that outputs the predicted accent class label for unseen test samples.
    def predict(self, X):
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        class_indices = np.argmax(raw_output, axis=1)
        return self.classes_[class_indices]

    # Probability estimation function that converts output decision scores into class probabilities using Softmax.
    def predict_proba(self, X):
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        exp_scores = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# ==============================================================================
# PIPELINE HELPER FUNCTIONS
# ==============================================================================

# Extracts all acoustic features from audio files once and caches them to a local CSV file.
# It saves execution time by preventing re-extraction of features during cross-validation loops.
def pre_extract_all_features_once(df_master):
    if EXTRACTED_FEATURES_MASTER_CSV.exists():
        print(f"──► [CACHED DATA DETECTED]: Loading features from {EXTRACTED_FEATURES_MASTER_CSV}")
        return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)

    print(f"\n⚡⚡⚡ [FIRST RUN INITIALIZATION]: Extracting acoustic features for {len(df_master)} files...")
    df_master_ml = df_master.copy()

    raw_normalized_dir = PROJECT_DATA_DIR / "audio_by_accent_raw_normalized"
    sentences_normalized_dir = PROJECT_DATA_DIR / "sentences_samples_normalized"

    missing_normalized_count = 0
    total_files = len(df_master_ml)

    # Internal helper function that locates and returns the corresponding normalized audio file path.
    def resolve_normalized_path(row):
        nonlocal missing_normalized_count
        accent_label = str(row['accent'])
        file_name = str(row['filename'])
        original_path = str(row['file_path'])

        if row.get('source_type') == 'archive' or 'sentence_' in original_path or 'sentences_samples' in original_path:
            part_match = [p for p in Path(original_path).parts if 'sentence_' in p]
            if part_match:
                normalized_path = sentences_normalized_dir / accent_label / part_match[0] / file_name
            else:
                normalized_path = sentences_normalized_dir / accent_label / file_name
        else:
            normalized_path = raw_normalized_dir / accent_label / file_name

        if normalized_path.exists():
            return str(normalized_path)
        else:
            missing_normalized_count += 1
            return original_path

    df_master_ml['file_path'] = df_master_ml.apply(resolve_normalized_path, axis=1)

    print(f"🔍 Normalized paths resolved: {total_files - missing_normalized_count}/{total_files}")
    temp_master_meta = WORKING_DIR / "temp_master_metadata_pipeline.csv"
    df_master_ml.to_csv(temp_master_meta, index=False, encoding='utf-8-sig')

    import_vector_to_file(str(temp_master_meta), str(raw_normalized_dir))

    default_output = WORKING_DIR / "accent_features_dataset.csv"
    if default_output.exists():
        if EXTRACTED_FEATURES_MASTER_CSV.exists():
            os.remove(EXTRACTED_FEATURES_MASTER_CSV)
        os.rename(default_output, EXTRACTED_FEATURES_MASTER_CSV)
        if temp_master_meta.exists():
            os.remove(temp_master_meta)
        print(f"──► [SUCCESS]: Saved features to {EXTRACTED_FEATURES_MASTER_CSV}")
    else:
        raise FileNotFoundError("Feature extraction subroutine failed.")

    return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)


# Merges master extracted features with current fold split metadata.
# It ensures exact alignment between acoustic features and target labels for train, validation, and test sets.
def filter_features_by_metadata_split(df_master_features, df_split_meta):
    return pd.merge(
        df_split_meta[['filename', 'file_path', 'speaker_id', 'text', 'accent']],
        df_master_features.drop(columns=['speaker_id', 'text', 'accent'], errors='ignore'),
        on='filename', how='inner'
    )


# Ranks and selects the top K most informative acoustic features using ANOVA F-tests on the training set.
# It prevents data leakage by selecting features strictly based on training fold data.
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


# Prepares, scales (StandardScaler), and encodes accent target labels for each cross-validation fold.
# It ensures models receive properly scaled and leakage-free feature matrices for training and evaluation.
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


# ==============================================================================
# HYPERPARAMETER TUNING ENGINE
# ==============================================================================

# Executes hyperparameter tuning or loads pre-defined custom parameters for classifiers inside inner CV folds.
# It helps find optimal model settings to achieve peak classification accuracy for each algorithm.
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


# ==============================================================================
# PLOTTING & REPORTING FUNCTIONS
# ==============================================================================

# Averages hyperparameter search accuracy across all folds and exports a summary CSV report.
# It documents the best-performing parameter configuration found for each classifier.
def export_and_print_best_hyperparameters(fold_tuning_history):
    if TUNING_MODE != "random_search" or not fold_tuning_history:
        return

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

        report_rows.append({
            'Classifier': model_name,
            'Best_Averaged_Accuracy_%': round(best_avg_accuracy, 2),
            'Optimal_Hyperparameters': str(best_param_dict)
        })

    df_best_report = pd.DataFrame(report_rows)
    report_csv_path = WORKING_DIR / f"best_hyperparameters_summary_{RUN_MODE}.csv"
    df_best_report.to_csv(report_csv_path, index=False)


# Generates side-by-side dashboard figures illustrating accuracy curves across parameter combinations.
# It provides a visual evaluation of hyperparameter sensitivity for each model.
def generate_hyperparameter_averaged_plots(fold_tuning_history):
    if TUNING_MODE != "random_search" or not fold_tuning_history:
        return

    for model_name, history in fold_tuning_history.items():
        if not history:
            continue

        flat_records = []
        for param_tuple, scores in history.items():
            param_dict = dict(param_tuple)
            param_dict['mean_accuracy'] = np.mean(scores) * 100
            flat_records.append(param_dict)

        df_params = pd.DataFrame(flat_records)
        if df_params.empty:
            continue

        best_row = df_params.loc[df_params['mean_accuracy'].idxmax()]

        if model_name == 'XGBoost':
            p_row, p_col = 'learning_rate', 'max_depth'
        elif model_name == 'Random Forest':
            p_row, p_col = 'n_estimators', 'max_depth'
        elif model_name in ['SVM (RBF)', 'Kernel ELM']:
            p_row, p_col = 'C', 'gamma'
        elif model_name == 'Logistic Regression':
            p_row, p_col = 'C', 'solver'
        elif model_name == 'Gaussian NB':
            p_row, p_col = 'var_smoothing', 'priors'
        else:
            p_row, p_col = df_params.columns[0], df_params.columns[1]

        df_params[p_row] = pd.to_numeric(df_params[p_row], errors='ignore')
        pivot_table = df_params.pivot_table(index=p_row, columns=p_col, values='mean_accuracy', aggfunc='mean')

        fig = plt.figure(figsize=(15, 7.5), dpi=300)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1], height_ratios=[0.12, 1, 0.18], hspace=0.35, wspace=0.25)

        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        banner_rect = plt.Rectangle((0, 0), 1, 1, facecolor='#1f4e78', edgecolor='none', transform=ax_title.transAxes)
        ax_title.add_patch(banner_rect)
        ax_title.text(0.5, 0.5, f"{model_name} Performance vs Hyperparameters ({RUN_MODE})",
                      color='white', weight='bold', fontsize=14, ha='center', va='center', transform=ax_title.transAxes)

        ax_table = fig.add_subplot(gs[1, 0])
        ax_table.axis('off')

        table_data = []
        col_headers = [f"{p_col}={col}" for col in pivot_table.columns]

        for idx_val, row_data in pivot_table.iterrows():
            formatted_row = [str(idx_val)]
            for val in row_data:
                formatted_row.append(f"{val:.2f}" if pd.notna(val) else "-")
            table_data.append(formatted_row)

        table = ax_table.table(
            cellText=table_data, colLabels=[p_row] + col_headers,
            cellLoc='center', loc='center', bbox=[0.0, 0.05, 1.0, 0.9]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.8)
            if r == 0:
                cell.set_facecolor('#d9e1f2')
                cell.set_text_props(weight='bold', color='#1f4e78')
            else:
                cell.set_facecolor('#f2f2f2' if r % 2 == 0 else '#ffffff')

        ax_line = fig.add_subplot(gs[1, 1])
        for col in pivot_table.columns:
            series = pivot_table[col].dropna()
            ax_line.plot(series.index.astype(str), series.values, marker='o', linewidth=1.8, markersize=5,
                         label=f"{p_col}={col}")

        ax_line.set_ylabel("Mean Accuracy (%)", fontweight='bold', fontsize=11)
        ax_line.set_xlabel(p_row, fontweight='bold', fontsize=11)
        ax_line.grid(True, linestyle='--', alpha=0.4)
        ax_line.legend(title=p_col, loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

        ax_box = fig.add_subplot(gs[2, :])
        ax_box.axis('off')
        box_rect = plt.Rectangle((0, 0), 1, 1, facecolor='#e2efda', edgecolor='#a9d08e', linewidth=1.2,
                                 transform=ax_box.transAxes)
        ax_box.add_patch(box_rect)

        display_params = [f"{p_row} = {best_row[p_row]}", f"{p_col} = {best_row[p_col]}"]
        best_str = ", ".join(display_params)
        ax_box.text(0.5, 0.62, "Best Hyperparameters:", fontsize=12, weight='bold', color='#375623', ha='center',
                    transform=ax_box.transAxes)
        ax_box.text(0.5, 0.25, best_str, fontsize=12, weight='bold', color='#1e3910', ha='center',
                    transform=ax_box.transAxes)

        clean_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        chart_path = GRAPHS_DIR / f"hyperparameter_impact_{clean_name}.png"
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()


# Creates bar charts comparing overall classifier accuracies and ANOVA feature selection frequencies.
# Updated to plot ALL Top 30 selected features instead of limiting to 15.
def generate_overall_plots(df_summary, df_feat_summary):
    # 1. Classifier Overall Accuracy Comparison
    plt.figure(figsize=(10, 5), dpi=300)
    models = df_summary['Classifier']
    acc_col = 'Overall Accuracy' if 'Overall Accuracy' in df_summary.columns else 'Mean Test Accuracy (%)'
    accuracies = [float(str(val).split('%')[0]) for val in df_summary[acc_col]]

    plt.bar(models, accuracies, color='#4c72b0', edgecolor='black', alpha=0.85)
    plt.title(f"Classifier Accuracy Comparison ({RUN_MODE})", fontsize=13, fontweight='bold')
    plt.xlabel("Classifier Model", fontsize=11, labelpad=10)
    plt.ylabel("Mean Test Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 1.5, f"{v:.2f}%", ha='center', fontweight='bold', fontsize=9)

    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    chart_path = GRAPHS_DIR / "classifier_accuracy_comparison.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Top 30 Selected Features Frequency Plot
    plt.figure(figsize=(10, 10), dpi=300)
    feat_col = df_feat_summary.columns[0]
    freq_col = df_feat_summary.columns[1]
    top_30_feats = df_feat_summary.head(30)

    plt.barh(top_30_feats[feat_col], top_30_feats[freq_col], color='#2ca02c', edgecolor='black', alpha=0.85)
    plt.xlabel("Selection Frequency (Out of 10 Folds)", fontsize=11)
    plt.ylabel("Acoustic Feature Name", fontsize=11)
    plt.title(f"Top 30 Features Selected by ANOVA ({RUN_MODE})", fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    feat_chart_path = GRAPHS_DIR / "top_features_frequency.png"
    plt.savefig(feat_chart_path, bbox_inches='tight', dpi=300)
    plt.close()


# Generates confusion matrix heatmaps showing true versus predicted accent classifications.
# It helps diagnose which specific accent classes are correctly predicted or commonly confused.
def generate_confusion_matrix_heatmaps(performance_tracker, class_labels):
    for model_name, tracker in performance_tracker.items():
        if not tracker['y_true_all']:
            continue

        cm = confusion_matrix(tracker['y_true_all'], tracker['y_pred_all'])
        plt.figure(figsize=(7, 5), dpi=300)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"Confusion Matrix - {model_name} ({RUN_MODE})", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Accent", fontsize=10)
        plt.ylabel("True Accent", fontsize=10)

        cm_path = GRAPHS_DIR / f"confusion_matrix_heatmap_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        plt.close()


# Creates violin plots showing feature value distributions across target accents for top ANOVA features.
# It illustrates how acoustic feature values differ between different accent categories.
def generate_feature_distribution_plots(df_master_features, top_features_list):
    top_5 = top_features_list[:5]
    for feat in top_5:
        if feat not in df_master_features.columns:
            continue

        plt.figure(figsize=(8, 4), dpi=300)
        sns.violinplot(x='accent', y=feat, data=df_master_features, palette='Set2', inner='quartile')
        plt.title(f"Feature Distribution across Accents: {feat}", fontsize=11, fontweight='bold')
        plt.xlabel("Accent Label", fontsize=10)
        plt.ylabel("Feature Value", fontsize=10)

        feat_dist_path = GRAPHS_DIR / f"feature_distribution_{feat}.png"
        plt.savefig(feat_dist_path, bbox_inches='tight', dpi=300)
        plt.close()


# Renders a 2D UMAP non-linear projection scatter plot of high-dimensional acoustic speech features.
# Fixed file saving mechanics to ensure reliable export directly into GRAPHS_DIR.
def generate_umap_plot(df_master_features):
    if umap is None:
        print("⚠️ Skipping UMAP plot (umap-learn module not installed).")
        return

    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_time',
                    'phonation_ratio', 'vowel_consonant_ratio', 'volume_max', 'volume_variance', 'volume_min',
                    'average_volume']
    feature_cols = [c for c in df_master_features.columns if c not in exclude_cols]

    X_raw = df_master_features[feature_cols].fillna(0)
    y_labels = df_master_features['accent'].astype(str).str.capitalize()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_labels, palette="Set1", alpha=0.7, s=55, ax=ax)
    ax.set_title(f"UMAP Acoustic Space Projection ({RUN_MODE})", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("UMAP Dimension 2", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title="Accent", loc='best', frameon=True)
    plt.tight_layout()

    umap_save_path = GRAPHS_DIR / "umap_accent_separability.png"
    fig.savefig(umap_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated UMAP plot -> {umap_save_path.name}")


# Computes and plots SHAP (Explainable AI) summary beeswarm figures for tree-based classifiers.
# Fixed figure context handling so individual accent SHAP beeswarms save cleanly to GRAPHS_DIR.
def generate_shap_plots(df_master_features):
    if shap is None:
        print("⚠️ Skipping SHAP plots (shap module not installed).")
        return

    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio',
                    'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    feature_cols = [c for c in df_master_features.columns if c not in exclude_cols]

    X_df = df_master_features[feature_cols].fillna(0)
    le = LabelEncoder()
    y_enc = le.fit_transform(df_master_features['accent'])

    xgb_shap_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                                   eval_metric='mlogloss', n_jobs=-1)
    xgb_shap_model.fit(X_df, y_enc)

    explainer = shap.TreeExplainer(xgb_shap_model)
    shap_values = explainer(X_df)

    for class_idx, class_name in enumerate(le.classes_):
        plt.close('all')
        fig = plt.figure(figsize=(10, 6), dpi=300)

        if len(shap_values.shape) == 3:
            vals_to_plot = shap_values.values[:, :, class_idx]
        else:
            vals_to_plot = shap_values[class_idx].values if isinstance(shap_values, list) else shap_values.values

        shap.summary_plot(vals_to_plot, X_df, max_display=12, show=False)
        plt.title(f"SHAP Feature Impact Profile: {class_name.capitalize()} Accent ({RUN_MODE})", fontsize=13,
                  fontweight='bold', pad=15)
        plt.xlabel("SHAP Value", fontsize=11)
        plt.tight_layout()

        shap_save_path = GRAPHS_DIR / f"shap_beeswarm_{class_name}.png"
        plt.savefig(shap_save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"✅ Generated SHAP Beeswarm plot -> {shap_save_path.name}")


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print(f"\n🚀 Starting Execution Pipeline...")

    # Load Dataset Metadata
    metadata_file_name = f"massive_metadata_{RUN_MODE}.csv"
    possible_paths = [
        PROJECT_DATA_DIR / f"machine_learning_official_{RUN_MODE}_only" / metadata_file_name,
        PROJECT_DATA_DIR / f"machine_learning_official_{RUN_MODE}" / metadata_file_name,
        PROJECT_DATA_DIR / metadata_file_name
    ]

    df_master = None
    for meta_path in possible_paths:
        if meta_path.exists():
            print(f"📂 [LOAD SUCCESS]: Found metadata at {meta_path}")
            df_master = pd.read_csv(meta_path)
            break

    if df_master is None:
        print("⚠️ Metadata CSV not found, building metadata dynamically...")
        if RUN_MODE == "free_text":
            df_master = build_free_text_only_metadata()
        elif RUN_MODE == "mixed":
            df_master = build_mixed_metadata()
        elif RUN_MODE == "archive_only":
            df_master = build_archive_only_metadata()

    if df_master is None or df_master.empty:
        raise ValueError("❌ Error: Master metadata dataframe is empty.")

    # Extract or load cached features
    df_master_features = pre_extract_all_features_once(df_master)

    # Verify zero speaker leakage across splits
    print("\n🔍 Running strict speaker leakage verification...")
    temp_fold_gen = run_nested_group_kfold_generator(df_master)
    for fold_i, train_m, val_m, test_m in temp_fold_gen:
        train_spks, val_spks, test_spks = set(train_m['speaker_id'].unique()), set(val_m['speaker_id'].unique()), set(
            test_m['speaker_id'].unique())
        if train_spks.intersection(val_spks) or train_spks.intersection(test_spks) or val_spks.intersection(test_spks):
            raise RuntimeError(f"❌ Speaker leakage detected in Fold {fold_i}!")
    print("✅ Zero speaker leakage verified!")

    # Model training and nested cross-validation
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
        print(f"\n--- Nested Cross-Validation Fold [{fold_idx}/10] ---")
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

                print(f"  [{model_name:<19}] Train Acc: {train_acc * 100:.2f}% | Test Acc: {overall_acc * 100:.2f}%")

        except Exception as e:
            print(f"Error in Fold {fold_idx}: {e}")
            traceback.print_exc()

    # Generate Aggregate Performance Summary
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
    print("\n================================================================================")
    print(f"FINAL CLASSIFICATION PERFORMANCE TABLE ({RUN_MODE.upper()} | {TUNING_MODE.upper()})")
    print("================================================================================")
    print(df_summary.to_string(index=False))

    # Feature Frequencies Summary Export
    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio',
                    'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    all_possible_features = [col for col in df_master_features.columns if col not in exclude_cols]
    feat_counts = Counter(all_selected_features_history)
    full_feat_counts = {feat: feat_counts.get(feat, 0) for feat in all_possible_features}

    df_feat_summary = pd.DataFrame(list(full_feat_counts.items()), columns=['Feature', 'Selection_Frequency_Out_Of_10'])
    df_feat_summary = df_feat_summary.sort_values(by=['Selection_Frequency_Out_Of_10', 'Feature'],
                                                  ascending=[False, True])

    out_csv_path = WORKING_DIR / f"top_selected_features_summary_{RUN_MODE}.csv"
    df_feat_summary.to_csv(out_csv_path, index=False)

    # 2D PCA Projection Plot Generation
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
        plt.title(f"2D PCA Projection of Top 30 Features ({RUN_MODE})\nVariance Explained: {total_exp_var:.2f}%",
                  fontsize=13, fontweight='bold', pad=15)
        plt.xlabel(f"PC1 ({exp_var_1:.2f}%)", fontsize=11, fontweight='bold')
        plt.ylabel(f"PC2 ({exp_var_2:.2f}%)", fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Accent", loc='best', frameon=True)
        plt.tight_layout()

        pca_top30_save_path = GRAPHS_DIR / "top_30_features_pca_2d.png"
        plt.savefig(pca_top30_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Generate and save all output figures directly into GRAPHS_DIR
    print("\n📊 Saving output graph figures to disk...")
    generate_overall_plots(df_summary, df_feat_summary)
    generate_hyperparameter_averaged_plots(fold_tuning_history)
    generate_confusion_matrix_heatmaps(performance_tracker, le.classes_)
    generate_feature_distribution_plots(df_master_features, df_feat_summary['Feature'].tolist())
    generate_umap_plot(df_master_features)
    generate_shap_plots(df_master_features)
    export_and_print_best_hyperparameters(fold_tuning_history)

    print("\n" + "=" * 80)
    print(f"✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   📂 Output Directory : {GRAPHS_DIR}")
    print("=" * 80 + "\n")