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
from scipy.spatial import ConvexHull

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RandomizedSearchCV, learning_curve
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

RUN_MODE = "free_text"  # Options: "free_text", "mixed", or "archive_only"
TUNING_MODE = "random_search"  # Options: "random_search", "custom", or "default"

ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 30

CUSTOM_HYPERPARAMETERS = {
    'Logistic Regression': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
    'SVM (RBF)': {'C': 10, 'gamma': 'scale'},
    'Random Forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    'XGBoost': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'Kernel ELM': {'C': 10, 'gamma': 0.01}
}

PROJECT_DIR = Path.cwd().resolve()
WORKING_DIR = PROJECT_DIR
PROJECT_DATA_DIR = PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

GRAPHS_DIR = PROJECT_DIR / "graphs" / RUN_MODE
if GRAPHS_DIR.exists():
    shutil.rmtree(GRAPHS_DIR)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f" PIPELINE EXECUTION CONFIGURATION:")
print(f"   • RUN_MODE     : '{RUN_MODE}'")
print(f"   • TUNING_MODE  : '{TUNING_MODE}'")
print(f"   • PROJECT_ROOT : {PROJECT_DIR}")
print(f"   • GRAPHS_DIR   : {GRAPHS_DIR} (Cleared & Prepared)")
print("=" * 80)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*max_iter.*')

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

K_fold.FREE_TEXT_OUT_DIR = WORKING_DIR / "machine_learning_official_free_text_only"
K_fold.MIXED_OUT_DIR = WORKING_DIR / "machine_learning_official_mixed"
K_fold.ARCHIVE_OUT_DIR = WORKING_DIR / "machine_learning_official_archive_only"

extracted_filename = f"master_extracted_features_{RUN_MODE}.csv"
EXTRACTED_FEATURES_MASTER_CSV = PROJECT_DIR / extracted_filename


# ==============================================================================
# CUSTOM CLASSIFIER DEFINITIONS
# ==============================================================================

class KernelELMClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn compatible estimator for Kernel Extreme Learning Machine (KELM).
    Provides analytical single-step training via RBF kernel mapping for fast high-dimensional learning.
    """
    def __init__(self, C=1.0, gamma=0.1, kernel='rbf'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

    def rbf_kernel(self, X1, X2):
        """Computes the Radial Basis Function (RBF) pairwise kernel matrix between feature matrices."""
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        distances_sq = np.maximum(distances_sq, 0.0)
        return np.exp(-self.gamma * distances_sq)

    def fit(self, X, y):
        """Fits the KELM weights analytically using regularized pseudoinverse inversion."""
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
        """Predicts class labels for unseen input sample feature vectors."""
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        class_indices = np.argmax(raw_output, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        """Calculates multi-class prediction probabilities using softmax output normalization."""
        X_data = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        K_test = self.rbf_kernel(X_data, self.X_train_)
        raw_output = np.dot(K_test, self.beta_)
        exp_scores = np.exp(raw_output - np.max(raw_output, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# ==============================================================================
# PIPELINE HELPER FUNCTIONS
# ==============================================================================

def pre_extract_all_features_once(df_master):
    """
    Extracts acoustic features for all audio samples in the metadata once and caches them to disk.
    Prevents redundant feature extraction during outer cross-validation loops.
    """
    if EXTRACTED_FEATURES_MASTER_CSV.exists():
        print(f"──► [CACHED DATA DETECTED]: Loading features from {EXTRACTED_FEATURES_MASTER_CSV}")
        return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)

    print(f"\n [FIRST RUN INITIALIZATION]: Extracting acoustic features for {len(df_master)} files...")
    df_master_ml = df_master.copy()

    raw_normalized_dir = PROJECT_DATA_DIR / "audio_by_accent_raw_normalized"
    sentences_normalized_dir = PROJECT_DATA_DIR / "sentences_samples_normalized"

    missing_normalized_count = 0
    total_files = len(df_master_ml)

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

    print(f" Normalized paths resolved: {total_files - missing_normalized_count}/{total_files}")
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


def filter_features_by_metadata_split(df_master_features, df_split_meta):
    """
    Filters and aligns cached feature vectors with a specific fold split metadata dataframe.
    """
    return pd.merge(
        df_split_meta[['filename', 'file_path', 'speaker_id', 'text', 'accent']],
        df_master_features.drop(columns=['speaker_id', 'text', 'accent'], errors='ignore'),
        on='filename', how='inner'
    )


def apply_anova_feature_selection(df_train, available_features, top_k=30):
    """
    Ranks acoustic features using ANOVA F-tests on the training fold and returns top K features.
    Guarantees strict leakage-free feature selection based solely on training split data.
    """
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
    """
    Prepares, scales (StandardScaler), and encodes target labels for train, validation, and test splits.
    Ensures zero data leakage across outer cross-validation folds.
    """
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

def tune_model_hyperparameters(model_name, X_train, y_train, fold_tuning_history, n_iter=20, cv_folds=3):
    """
    Performs RandomizedSearchCV hyperparameter tuning on training fold data and tracks history.
    """
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

def export_and_print_best_hyperparameters(fold_tuning_history):
    """
    Averages hyperparameter validation accuracy scores across outer folds for ALL 6 classifiers and exports CSV.
    """
    if TUNING_MODE != "random_search" or not fold_tuning_history:
        print(" Hyperparameter report skipped.")
        return

    print("\n" + "=" * 80)
    print(" OPTIMAL HYPERPARAMETERS REPORT (AVERAGED ACROSS ALL FOLDS)")
    print("=" * 80)

    report_rows = []
    for model_name in ['Gaussian NB', 'Logistic Regression', 'SVM (RBF)', 'Random Forest', 'XGBoost', 'Kernel ELM']:
        history = fold_tuning_history.get(model_name, {})
        if not history:
            print(f"\n Classifier: [{model_name}] -> No tuning history recorded.")
            continue

        best_param_dict = None
        best_avg_accuracy = -1.0

        for param_tuple, scores in history.items():
            if not scores:
                continue
            avg_score = np.mean(scores) * 100
            if avg_score > best_avg_accuracy:
                best_avg_accuracy = avg_score
                best_param_dict = dict(param_tuple)

        if best_param_dict is not None:
            print(f"\n Classifier: [{model_name}]")
            print(f"   • Mean Cross-Validation Accuracy: {best_avg_accuracy:.2f}%")
            print(f"   • Best Hyperparameters:")
            for k, v in best_param_dict.items():
                print(f"       - {k}: {v}")

            report_rows.append({
                'Classifier': model_name,
                'Best_Averaged_Accuracy_%': round(best_avg_accuracy, 2),
                'Optimal_Hyperparameters': str(best_param_dict)
            })

    if report_rows:
        df_best_report = pd.DataFrame(report_rows)
        report_csv_path = WORKING_DIR / f"best_hyperparameters_summary_{RUN_MODE}.csv"
        df_best_report.to_csv(report_csv_path, index=False)
        print(f"\n Summary report saved -> {report_csv_path}")


def generate_hyperparameter_averaged_plots(fold_tuning_history):
    """
    Generates combined (Table on left, Continuous sensitivity curve on right) dashboard figures for ALL 6 models.
    """
    if TUNING_MODE != "random_search" or not fold_tuning_history:
        return

    print("==================================================")
    print(" GENERATING COMBINED (TABLE + CHART) FIGURES FOR ALL 6 CLASSIFIERS")
    print("==================================================")

    for model_name, history in fold_tuning_history.items():
        if not history:
            continue

        flat_records = []
        for param_tuple, scores in history.items():
            param_dict = dict(param_tuple)
            mean_acc = np.mean(scores) * 100
            param_dict['Accuracy'] = mean_acc
            flat_records.append(param_dict)

        df_model = pd.DataFrame(flat_records)
        if df_model.empty:
            continue

        x_param = None
        series_param = None

        if model_name in ['Kernel ELM', 'SVM (RBF)']:
            x_param = 'C'
            series_param = 'gamma'
        elif model_name == 'Logistic Regression':
            x_param = 'C'
            series_param = 'solver'
        elif model_name == 'Random Forest':
            x_param = 'n_estimators'
            series_param = 'max_depth'
        elif model_name == 'XGBoost':
            x_param = 'learning_rate'
            series_param = 'max_depth'
        elif model_name == 'Gaussian NB':
            x_param = 'var_smoothing'
            series_param = 'priors' if 'priors' in df_model.columns else None

        if not x_param or x_param not in df_model.columns:
            param_cols = [c for c in df_model.columns if c != 'Accuracy']
            if param_cols:
                x_param = param_cols[0]
                series_param = param_cols[1] if len(param_cols) > 1 else None

        df_model[x_param] = pd.to_numeric(df_model[x_param], errors='coerce')
        df_model = df_model.dropna(subset=[x_param]).sort_values(by=x_param)

        if series_param and series_param in df_model.columns:
            df_model[series_param] = df_model[series_param].apply(lambda x: str(x) if x is not None else 'Default/None')
            pivot_df = df_model.pivot_table(index=x_param, columns=series_param, values='Accuracy', aggfunc='mean')
        else:
            pivot_df = df_model.groupby(x_param)[['Accuracy']].mean()

        clean_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')

        fig, (ax_table, ax_plot) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]}, dpi=300)

        ax_table.axis('off')
        table_data = pivot_df.round(2).fillna("-").astype(str).values
        col_labels = [f"{series_param}={c}" if series_param else str(c) for c in pivot_df.columns]
        row_labels = [f"{x_param}={r}" for r in pivot_df.index]

        table_obj = ax_table.table(
            cellText=table_data, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center'
        )
        table_obj.auto_set_font_size(False)
        table_obj.set_fontsize(10)
        table_obj.scale(1.1, 1.8)
        ax_table.set_title(f"{model_name} - Hyperparameter Grid (%)", fontsize=12, fontweight='bold', pad=10)

        for col in pivot_df.columns:
            series_data = pivot_df[col].dropna()
            if not series_data.empty:
                label_name = f"{series_param}={col}" if series_param else col
                ax_plot.plot(series_data.index, series_data.values, marker='o', linewidth=2, markersize=6, label=label_name)

        if x_param in ['C', 'var_smoothing']:
            ax_plot.set_xscale('log')

        ax_plot.set_title(f"{model_name} - Performance Curve", fontsize=12, fontweight='bold', pad=10)
        ax_plot.set_xlabel(f"{x_param} Parameter", fontsize=11)
        ax_plot.set_ylabel("Mean Accuracy (%)", fontsize=11)
        ax_plot.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_plot.legend(title=series_param if series_param else "", fontsize=9, loc='best')

        plt.tight_layout()

        combined_image_path = GRAPHS_DIR / f"hyperparameter_impact_{clean_name}.png"
        plt.savefig(combined_image_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Generated combined figure for [{model_name}] -> {combined_image_path.name}")


def generate_overall_plots(df_summary, df_feat_summary):
    """
    Renders overall accuracy comparison bar chart and Top 30 ANOVA selected feature frequency chart.
    """
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


def generate_confusion_matrix_heatmaps(performance_tracker, class_labels):
    """
    Renders and exports Seaborn confusion matrix heatmaps for all tested machine learning models.
    """
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


def generate_umap_plot(df_master_features):
    """
    Computes 2D UMAP non-linear manifold projection of acoustic speech feature space across target accents.
    """
    if umap is None:
        print(" Skipping UMAP plot (umap-learn module not installed).")
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
    print(f" Generated UMAP plot -> {umap_save_path.name}")


def generate_shap_plots(df_master_features):
    """
    Generates SHAP (Explainable AI) summary beeswarm plots for tree-based ensemble classifiers per accent.
    """
    if shap is None:
        print(" Skipping SHAP plots (shap module not installed).")
        return

    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio',
                    'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    feature_cols = [c for c in df_master_features.columns if c not in exclude_cols]

    X_df = df_master_features[feature_cols].fillna(0)
    le = LabelEncoder()
    y_enc = le.fit_transform(df_master_features['accent'])

    xgb_shap_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss', n_jobs=-1)
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
        plt.title(f"SHAP Feature Impact Profile: {class_name.capitalize()} Accent ({RUN_MODE})", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("SHAP Value", fontsize=11)
        plt.tight_layout()

        shap_save_path = GRAPHS_DIR / f"shap_beeswarm_{class_name}.png"
        plt.savefig(shap_save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f" Generated SHAP Beeswarm plot -> {shap_save_path.name}")


def generate_vsa_plot(df_master_features):
    """
    Computes Vowel Space Area (VSA) Convex Hull polygons in inverted F1/F2 formant space across all accent groups.
    """
    f1_col = [c for c in df_master_features.columns if 'F1' in c or 'f1' in c]
    f2_col = [c for c in df_master_features.columns if 'F2' in c or 'f2' in c]

    if not (f1_col and f2_col):
        print(" Could not locate Formant columns (F1/F2) for VSA plot.")
        return

    f1_name, f2_name = f1_col[0], f2_col[0]
    color_palette = {'chinese': '#ff7f0e', 'indian': '#e41a1c', 'nigerian': '#2ca02c', 'usa': '#377eb8'}

    plt.figure(figsize=(10, 8), dpi=300)
    vsa_summary = {}

    for acc in df_master_features['accent'].unique():
        acc_clean = str(acc).lower()
        sub_df = df_master_features[df_master_features['accent'] == acc][[f2_name, f1_name]].dropna()
        points = sub_df.values

        if len(points) >= 3:
            hull = ConvexHull(points)
            vsa_area = hull.volume
            vsa_summary[acc_clean] = vsa_area
            c_color = color_palette.get(acc_clean, '#333333')

            hull_pts = np.append(hull.vertices, hull.vertices[0])
            plt.plot(points[hull_pts, 0], points[hull_pts, 1], linestyle='--', linewidth=2, color=c_color, label=f"{acc_clean.capitalize()} (VSA = {vsa_area:,.0f} Hz²)")
            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=c_color, alpha=0.12)
            plt.scatter(points[:, 0], points[:, 1], color=c_color, alpha=0.35, s=30)

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.title("Unified Vowel Space Area (VSA) Comparison Across All Accents\n(Convex Hull Geometry in F1/F2 Space)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("F2 Frequency (Hz) [Inverted - Tongue Back/Front]", fontsize=11)
    plt.ylabel("F1 Frequency (Hz) [Inverted - Tongue Height]", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Accent Group & VSA Hull Area", fontsize=10, loc='best')
    plt.tight_layout()

    vsa_path = GRAPHS_DIR / "unified_vowel_space_area_all_accents.png"
    plt.savefig(vsa_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n Vowel Space Area (VSA) Summary for All Accents:")
    for acc_name, area_val in sorted(vsa_summary.items()):
        print(f"   ├─ {acc_name.capitalize():<10}: {area_val:,.2f} Hz²")
    print(f" Generated VSA plot -> {vsa_path.name}")


def generate_learning_curves_and_gap_report(performance_tracker, df_master_features):
    """
    Outputs Train vs Test diagnostic accuracy gap table and renders learning curves to diagnose overfitting.
    """
    print("\n" + "=" * 75)
    print(" CLASSIFIER TRAIN VS. TEST ACCURACY & OVERFITTING GAP TABLE")
    print("=" * 75)

    gap_summary_rows = []
    for model_name, tracker in performance_tracker.items():
        if tracker['train_acc'] and tracker['overall']:
            mean_train_acc = np.mean(tracker['train_acc']) * 100
            mean_test_acc = np.mean(tracker['overall']) * 100
            gap = mean_train_acc - mean_test_acc

            gap_summary_rows.append({
                'Classifier': model_name,
                'Mean Train Accuracy (%)': f"{mean_train_acc:.2f}%",
                'Mean Test Accuracy (%)': f"{mean_test_acc:.2f}%",
                'Overfitting / Generalization Gap (%)': f"{gap:.2f}%"
            })

    df_gap_summary = pd.DataFrame(gap_summary_rows)
    print(df_gap_summary.to_string(index=False))
    print("=" * 75 + "\n")

    def plot_model_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8)):
        plt.figure(figsize=(9, 5), dpi=300)
        plt.title(f"Learning Curve ({title}) - Overfitting Diagnosis", fontsize=12, fontweight='bold')
        plt.xlabel("Training Examples Count", fontsize=11)
        plt.ylabel("Accuracy Score (%)", fontsize=11)

        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy')

        train_scores_mean = np.mean(train_scores, axis=1) * 100
        train_scores_std = np.std(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100
        test_scores_std = np.std(test_scores, axis=1) * 100

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score (Train Acc)")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score (Test Acc)")

        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()

        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '')
        save_path = GRAPHS_DIR / f"learning_curve_{clean_title}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved Learning Curve diagnostic plot -> {save_path.name}")

    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio', 'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    feature_cols = [c for c in df_master_features.columns if c not in exclude_cols]

    X_diag = df_master_features[feature_cols].fillna(0).values
    le_diag = LabelEncoder()
    y_diag = le_diag.fit_transform(df_master_features['accent'])

    xgb_diag = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    plot_model_learning_curve(xgb_diag, "XGBoost", X_diag, y_diag)

    rf_diag = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    plot_model_learning_curve(rf_diag, "Random Forest", X_diag, y_diag)


# ==============================================================================
# MAIN PIPELINE EXECUTION & SINGLE-FILE INFERENCE
# ==============================================================================

if __name__ == "__main__":
    print(f"\n Starting Execution Pipeline...")

    metadata_file_name = f"massive_metadata_{RUN_MODE}.csv"
    possible_paths = [
        PROJECT_DATA_DIR / f"machine_learning_official_{RUN_MODE}_only" / metadata_file_name,
        PROJECT_DATA_DIR / f"machine_learning_official_{RUN_MODE}" / metadata_file_name,
        PROJECT_DATA_DIR / metadata_file_name
    ]

    df_master = None
    for meta_path in possible_paths:
        if meta_path.exists():
            print(f" [LOAD SUCCESS]: Found metadata at {meta_path}")
            df_master = pd.read_csv(meta_path)
            break

    if df_master is None:
        print(" Metadata CSV not found, building metadata dynamically...")
        if RUN_MODE == "free_text":
            df_master = build_free_text_only_metadata()
        elif RUN_MODE == "mixed":
            df_master = build_mixed_metadata()
        elif RUN_MODE == "archive_only":
            df_master = build_archive_only_metadata()

    if df_master is None or df_master.empty:
        raise ValueError(" Error: Master metadata dataframe is empty.")

    df_master_features = pre_extract_all_features_once(df_master)

    print("\n Running strict speaker leakage verification...")
    temp_fold_gen = run_nested_group_kfold_generator(df_master)
    for fold_i, train_m, val_m, test_m in temp_fold_gen:
        train_spks, val_spks, test_spks = set(train_m['speaker_id'].unique()), set(val_m['speaker_id'].unique()), set(test_m['speaker_id'].unique())
        if train_spks.intersection(val_spks) or train_spks.intersection(test_spks) or val_spks.intersection(test_spks):
            raise RuntimeError(f" Speaker leakage detected in Fold {fold_i}!")
    print(" Zero speaker leakage verified!")

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

                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
                for acc_label in accents:
                    if acc_label in report:
                        performance_tracker[model_name]['per_accent'][acc_label].append(report[acc_label]['recall'])

                print(f"    [{model_name:<19}] Test Acc: {overall_acc * 100:.2f}%")

        except Exception as e:
            print(f"Error in Fold {fold_idx}: {e}")
            traceback.print_exc()

    summary_rows = []
    for model_name in model_names:
        tracker = performance_tracker[model_name]
        if not tracker['overall']:
            continue

        mean_overall = np.mean(tracker['overall']) * 100
        std_overall = np.std(tracker['overall']) * 100

        row = {
            'Classifier': model_name,
            'Overall Accuracy': f"{mean_overall:.2f}% ± {std_overall:.2f}%"
        }
        for acc_label in accents:
            scores = tracker['per_accent'][acc_label]
            row[f"{acc_label.capitalize()} Accuracy"] = f"{np.mean(scores) * 100:.2f}%" if scores else "N/A"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\n================================================================================")
    print("FINAL ACCENT CLASSIFICATION PERFORMANCE TABLE (TEST ACCURACY & PER-ACCENT SCORES)")
    print("================================================================================")
    print(df_summary.to_string(index=False))

    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio', 'volume_max', 'volume_variance', 'volume_min', 'average_volume']
    all_possible_features = [col for col in df_master_features.columns if col not in exclude_cols]
    feat_counts = Counter(all_selected_features_history)
    full_feat_counts = {feat: feat_counts.get(feat, 0) for feat in all_possible_features}

    df_feat_summary = pd.DataFrame(list(full_feat_counts.items()), columns=['Feature', 'Selection_Frequency_Out_Of_10'])
    df_feat_summary = df_feat_summary.sort_values(by=['Selection_Frequency_Out_Of_10', 'Feature'], ascending=[False, True])

    out_csv_path = WORKING_DIR / f"top_selected_features_summary_{RUN_MODE}.csv"
    df_feat_summary.to_csv(out_csv_path, index=False)

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
        plt.title(f"2D PCA Projection of Top 30 Features ({RUN_MODE})\nVariance Explained: {total_exp_var:.2f}%", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel(f"PC1 ({exp_var_1:.2f}%)", fontsize=11, fontweight='bold')
        plt.ylabel(f"PC2 ({exp_var_2:.2f}%)", fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Accent", loc='best', frameon=True)
        plt.tight_layout()

        pca_top30_save_path = GRAPHS_DIR / "top_30_features_pca_2d.png"
        plt.savefig(pca_top30_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("\n Generating and saving all output graph figures...")
    generate_overall_plots(df_summary, df_feat_summary)
    generate_hyperparameter_averaged_plots(fold_tuning_history)
    generate_confusion_matrix_heatmaps(performance_tracker, le.classes_)
    generate_umap_plot(df_master_features)
    generate_shap_plots(df_master_features)
    generate_vsa_plot(df_master_features)
    export_and_print_best_hyperparameters(fold_tuning_history)
    generate_learning_curves_and_gap_report(performance_tracker, df_master_features)

    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"    Output Directory : {GRAPHS_DIR}")
    print("=" * 80 + "\n")

    # ==============================================================================
    # DYNAMIC BEST MODEL TRAINING & SINGLE-FILE INFERENCE (PYCHARM TKINTER DIALOG)
    # ==============================================================================

    # 1. Dynamically Select Absolute Best Performing Classifier from Outer Folds
    model_means = {}
    for m in performance_tracker:
        if performance_tracker[m]['overall']:
            model_means[m] = np.mean(performance_tracker[m]['overall']) * 100

    if not model_means:
        raise RuntimeError(" Error: No classification results found in performance_tracker!")

    best_model_name = max(model_means, key=model_means.get)
    highest_accuracy = model_means[best_model_name]

    print("=" * 80)
    print(" CROSS-VALIDATION WINNING MODEL SELECTION SUMMARY:")
    for m, acc in sorted(model_means.items(), key=lambda item: item[1], reverse=True):
        marker = " [SELECTED FOR INFERENCE]" if m == best_model_name else ""
        print(f"   ├─ {m:<19}: {acc:.2f}% {marker}")
    print("=" * 80)
    print(f" Dynamic Best Classifier Selected: [{best_model_name}] ({highest_accuracy:.2f}% Mean Test Acc)")
    print("=" * 80)

    # 2. Select Features (Dynamically based on ENABLE_FEATURE_SELECTION)
    # Checks whether ANOVA feature selection was enabled or disabled in configuration.
    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent', 'total_duration', 'phonation_ratio', 'volume_max', 'volume_variance', 'volume_min', 'average_volume']

    if 'ENABLE_FEATURE_SELECTION' in globals() and ENABLE_FEATURE_SELECTION and 'df_feat_summary' in globals():
        # Feature selection enabled: use top selected features from ANOVA
        k_features = TOP_K_FEATURES if 'TOP_K_FEATURES' in globals() else 30
        top_selected_features = [f for f in df_feat_summary.head(k_features)['Feature'].tolist() if f not in exclude_cols]
        print(f" Feature Selection: ENABLED (Using Top {len(top_selected_features)} ANOVA Features)")
    else:
        # Feature selection disabled: use all available acoustic features
        top_selected_features = [col for col in df_master_features.columns if col not in exclude_cols]
        print(f" Feature Selection: DISABLED (Using ALL {len(top_selected_features)} Acoustic Features)")

    X_full = df_master_features[top_selected_features].fillna(0).values

    le_final = LabelEncoder()
    y_full = le_final.fit_transform(df_master_features['accent'])

    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)

    # 3. Instantiate and Fit Winning Classifier Object
    def get_winning_classifier_instance(model_name):
        """Returns fresh estimator instance of winning classifier model."""
        if model_name == 'Gaussian NB':
            return GaussianNB(priors=[0.25, 0.25, 0.25, 0.25])
        elif model_name == 'Logistic Regression':
            return LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced')
        elif model_name == 'SVM (RBF)':
            return SVC(probability=True, random_state=42, class_weight='balanced')
        elif model_name == 'Random Forest':
            return RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
        elif model_name == 'XGBoost':
            return XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
        elif model_name == 'Kernel ELM':
            return KernelELMClassifier()
        else:
            raise ValueError(f"Unknown classifier model name: {model_name}")

    best_model_obj = get_winning_classifier_instance(best_model_name)

    if fold_tuning_history.get(best_model_name):
        best_param_dict = dict(max(fold_tuning_history[best_model_name].items(), key=lambda item: np.mean(item[1]))[0])
        if hasattr(best_model_obj, 'set_params'):
            if hasattr(best_model_obj, 'get_params') and 'class_weight' in best_model_obj.get_params():
                best_param_dict['class_weight'] = 'balanced'
            best_model_obj.set_params(**best_param_dict)

    best_model_obj.fit(X_full_scaled, y_full)
    print(f" Dynamic Winner [{best_model_name}] Trained Successfully on Full Dataset Features!\n")

    # 4. Process and Classify Uploaded Audio File Pipeline
    def process_and_classify_audio_pipeline_exact(file_path):
        """
        Executes exact dataset preprocessing (load, trim, RMS normalize), extracts features,
        and predicts accent label with confidence probabilities for an individual audio sample file.
        """
        uploaded_path = Path(file_path)
        print(f"\n Processing audio via Exact Dataset Pipeline: {uploaded_path.name}...")

        try:
            y, sr = librosa.load(uploaded_path, sr=16000, mono=True)
            if len(y) / sr < 0.1:
                print(" Error: File too short.")
                return

            y_trimmed, _ = librosa.effects.trim(y, top_db=40)
            if len(y_trimmed) / sr < 0.1:
                print(" Error: File too short after trimming.")
                return

            rms = np.sqrt(np.mean(y_trimmed ** 2))
            if rms > 0:
                target_rms = 0.05
                y_normalized = y_trimmed * (target_rms / rms)
                y_normalized = np.clip(y_normalized, -0.99, 0.99)
            else:
                print(" Error: Invalid RMS Energy.")
                return

            norm_path = WORKING_DIR / f"norm_{uploaded_path.name}"
            sf.write(norm_path, y_normalized, 16000)

            temp_single_meta = WORKING_DIR / "temp_single_file_metadata.csv"
            df_single = pd.DataFrame([{'filename': norm_path.name, 'file_path': str(norm_path.resolve()), 'speaker_id': 'user', 'accent': 'unknown', 'text': 'sample'}])
            df_single.to_csv(temp_single_meta, index=False)

            import_vector_to_file(str(temp_single_meta), str(norm_path.parent))
            extracted_csv = WORKING_DIR / "accent_features_dataset.csv"

            if not extracted_csv.exists():
                print(" Error: Feature extraction failed.")
                return

            df_single_feat = pd.read_csv(extracted_csv)

            if temp_single_meta.exists(): os.remove(temp_single_meta)
            if extracted_csv.exists(): os.remove(extracted_csv)
            if norm_path.exists(): os.remove(norm_path)

            for col in top_selected_features:
                if col not in df_single_feat.columns:
                    df_single_feat[col] = 0.0

            X_single = df_single_feat[top_selected_features].fillna(0).values
            X_single_scaled = final_scaler.transform(X_single)

            if hasattr(best_model_obj, "predict_proba"):
                probs = best_model_obj.predict_proba(X_single_scaled)[0]
                top_class_idx = np.argmax(probs)
                top_accent_label = le_final.inverse_transform([top_class_idx])[0]

                print("\nResults breakdown:")
                for cls_name, prob in zip(le_final.classes_, probs):
                    print(f"    ├─ {cls_name.capitalize():<10}: {prob * 100:.2f}%")
            else:
                top_class_idx = best_model_obj.predict(X_single_scaled)[0]
                top_accent_label = le_final.inverse_transform([top_class_idx])[0]

            print("=" * 60)
            print(f"YOUR ACCENT IS : {top_accent_label.upper()}")
            print("=" * 60)

        except Exception as e:
            print(f"Error during single-file inference: {e}")
            traceback.print_exc()

    # 5. Open System File Dialog for User Selection (Way 3 via Tkinter)
    print("\n Opening file browser dialog to select audio WAV file for inference...")
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        selected_audio_path = filedialog.askopenfilename(
            title="Select Audio WAV File for Accent Inference",
            filetypes=[("Audio Files", "*.wav")]
        )
        root.destroy()

        if selected_audio_path:
            process_and_classify_audio_pipeline_exact(selected_audio_path)
        else:
            print(" File selection cancelled by user.")
    except Exception as dialog_err:
        print(f"warning: Could not open file dialog window ({dialog_err}).")
        user_file = input("Please manually enter the full path to your audio WAV file: ")
        if user_file and os.path.exists(user_file):
            process_and_classify_audio_pipeline_exact(user_file)