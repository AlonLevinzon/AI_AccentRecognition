import os
import sys
import pandas as pd
import numpy as np
import warnings
import traceback
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RandomizedSearchCV

# Ignore benign runtime warnings and deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import all classifiers directly from your project setup
from classifiers.GNB import get_gaussian_nb_classifier
from classifiers.KELM import KernelELMClassifier
from classifiers.logistic_regression_classifier import get_logistic_regression_classifier
from classifiers.random_forest_classifier import get_random_forest_classifier
from classifiers.svm_classifier import get_svm_classifier
from classifiers.XGBoost import get_xgboost_classifier

# Import core pipeline components from K_fold
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

# =====================================================================
# GLOBAL CONFIGURATION FOR SPEED OPTIMIZATION & FEATURE SELECTION
# =====================================================================
RUN_MODE = "archive_only"  # Toggle between "free_text", "mixed", or "archive_only"

# Feature Selection Settings
ENABLE_FEATURE_SELECTION = False
TOP_K_FEATURES = 25  # Number of top acoustic features to retain via ANOVA

# Hyperparameter Optimization Modes: "random_search", "custom", or "default"
TUNING_MODE = "default"

# Custom Parameters Dictionary (Used when TUNING_MODE = "custom")
CUSTOM_HYPERPARAMETERS = {
    'Logistic Regression': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
    'SVM (RBF)': {'C': 10, 'gamma': 'scale'},
    'Random Forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1,
                      'max_features': 'sqrt'},
    'XGBoost': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'Kernel ELM': {'C': 10, 'gamma': 0.01}
}

current_dir = Path(__file__).resolve().parent
EXTRACTED_FEATURES_MASTER_CSV = current_dir / f"master_extracted_features_{RUN_MODE}.csv"


def build_archive_only_metadata():
    """
    Pipeline for archive_only mode: Mimics the exact extraction logic from K_fold's mixed mode Part 2.
    """
    print("\n--- Building Archive Only Massive Metadata ---")
    all_records = []
    accents = ['chinese', 'indian', 'nigerian', 'usa']
    sentences_dir = current_dir / "sentences_samples_normalized"
    archive_out_dir = current_dir / "machine_learning_official_archive_only"

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

        speaker_col_name = None
        for key in ['speakerid', 'speaker_id', 'client_id', 'id']:
            if key in df_raw.columns:
                speaker_col_name = key
                break

        df_unique_speakers = df_raw.copy()
        if speaker_col_name:
            df_unique_speakers['temp_spk'] = df_unique_speakers[speaker_col_name].apply(clean_speaker_id)
            df_unique_speakers = df_unique_speakers.drop_duplicates(subset=['temp_spk'])

        sentence_records = []
        for idx, row in df_unique_speakers.iterrows():
            raw_filename = ""
            for key in ['filename', 'file_name', 'audio', 'file', 'wav_path']:
                if key in df_unique_speakers.columns:
                    val = df_unique_speakers.at[idx, key]
                    if pd.notna(val):
                        raw_filename = val
                        break

            if not raw_filename:
                continue

            pure_filename = clean_filename_path(raw_filename)

            speaker_raw = ""
            for key in ['speakerid', 'speaker_id', 'client_id', 'id']:
                if key in df_unique_speakers.columns:
                    val = df_unique_speakers.at[idx, key]
                    if pd.notna(val):
                        speaker_raw = val
                        break
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
                        'filename': archive_filename,
                        'file_path': str(valid_path),
                        'speaker_id': speaker_id,
                        'accent': accent_label,
                        'text': final_text,
                        'source_type': 'archive'
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
    """
    Loads pre-extracted features from EXTRACTED_FEATURES_MASTER_CSV if it exists.
    """
    if EXTRACTED_FEATURES_MASTER_CSV.exists():
        print(f"──► [CACHED DATA DETECTED]: Loading pre-extracted features from {EXTRACTED_FEATURES_MASTER_CSV}")
        return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)

    print(f"\n⚡⚡⚡ [FIRST RUN INITIALIZATION]: Extracting acoustic features for ALL {len(df_master)} files once...")
    temp_master_meta = current_dir / "temp_master_metadata_pipeline.csv"
    df_master.to_csv(temp_master_meta, index=False, encoding='utf-8-sig')

    import_vector_to_file(str(temp_master_meta), str(current_dir / "audio_by_accent_raw_normalized"))

    default_output = Path("accent_features_dataset.csv")
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
    """
    Merges master feature dataset with fold split metadata via RAM.
    """
    df_merged = pd.merge(
        df_split_meta[['filename', 'file_path', 'speaker_id', 'text', 'accent']],
        df_master_features.drop(columns=['speaker_id', 'text', 'accent'], errors='ignore'),
        on='filename',
        how='inner'
    )
    return df_merged


def apply_anova_feature_selection(df_train, available_features, top_k=25):
    """
    LEAK-FREE FEATURE SELECTION: Filters zero-variance features first,
    then computes ANOVA F-Scores strictly on the training fold.
    """
    X_tr = df_train[available_features].fillna(0)

    # Filter out constant features (variance == 0) to prevent division by zero warnings
    non_constant_features = [col for col in available_features if X_tr[col].nunique() > 1]

    if not non_constant_features:
        return available_features[:top_k]

    X_filtered = X_tr[non_constant_features]
    y_tr = df_train['accent']

    f_scores, _ = f_classif(X_filtered, y_tr)
    anova_df = pd.DataFrame({'Feature': non_constant_features, 'F_Score': f_scores})

    # Replace NaNs in f_scores with 0
    anova_df['F_Score'] = anova_df['F_Score'].fillna(0)

    selected_features = anova_df.sort_values(by='F_Score', ascending=False)['Feature'].head(top_k).tolist()
    return selected_features


def tune_model_hyperparameters(model_name, X_train, y_train, fold_tuning_history, n_iter=60, cv_folds=3):
    """
    Wrapper function supporting all scikit-learn compatible classifiers.
    Silently records parameter combinations to calculate global fold averages.
    """
    if TUNING_MODE == "default":
        return None, {}

    if TUNING_MODE == "custom":
        custom_params = CUSTOM_HYPERPARAMETERS.get(model_name, {})
        print(f"      🔧 [CUSTOM MODE]: Applied user-defined parameters for [{model_name}]: {custom_params}")

        if model_name == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42, **custom_params), custom_params
        elif model_name == 'SVM (RBF)':
            from sklearn.svm import SVC
            return SVC(probability=True, random_state=42, **custom_params), custom_params
        elif model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42, n_jobs=-1, **custom_params), custom_params
        elif model_name == 'XGBoost':
            from xgboost import XGBClassifier
            return XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1, **custom_params), custom_params
        elif model_name == 'Kernel ELM':
            return KernelELMClassifier(**custom_params), custom_params
        return None, {}

    if TUNING_MODE == "random_search":
        if model_name == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            base_model = LogisticRegression(max_iter=1000, random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga', 'newton-cg']
            }

        elif model_name == 'SVM (RBF)':
            from sklearn.svm import SVC
            base_model = SVC(probability=True, random_state=42)
            param_grid = {
                'C': [0.1, 1, 10, 50, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }

        elif model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

        elif model_name == 'XGBoost':
            from xgboost import XGBClassifier
            base_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }
        else:
            return None, {}

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        # Record parameter search results for cross-fold averaging
        for params, mean_score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score']):
            param_key = tuple(sorted(params.items()))
            fold_tuning_history[model_name][param_key].append(mean_score)

        return search.best_estimator_, search.best_params_

    return None, {}


def load_and_scale_fold_matrices(df_master_features, df_train_meta, df_val_meta, df_test_meta):
    """
    Prepares train, validation, and test matrices with leak-free ANOVA selection and scaling.
    """
    df_train = filter_features_by_metadata_split(df_master_features, df_train_meta)
    df_val = filter_features_by_metadata_split(df_master_features, df_val_meta)
    df_test = filter_features_by_metadata_split(df_master_features, df_test_meta)

    # Exclude metadata and biased/leaked metrics
    exclude_cols = [
        'filename', 'file_path', 'speaker_id', 'text', 'accent',
        'total_duration', 'phonation_ratio', 'volume_max', 'volume_variance'
    ]
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


def export_and_print_best_hyperparameters(fold_tuning_history):
    """
    Generates a dedicated summary report (CSV) and prints a clean, classifier-specific
    breakdown of optimal hyperparameters averaged across ALL 10 folds.
    """
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

        # Print per-classifier breakdown to console
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

    # Save dedicated CSV report
    df_best_report = pd.DataFrame(report_rows)
    report_csv_path = current_dir / "best_hyperparameters_summary.csv"
    df_best_report.to_csv(report_csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"💾 Dedicated Hyperparameters Report exported successfully -> {report_csv_path}")
    print("=" * 80)


def generate_hyperparameter_averaged_plots(fold_tuning_history):
    """
    Generates and saves averaged hyperparameter performance plots for ALL tuned classifiers across all folds.
    Fixes non-numeric string sorting on X-axis to eliminate zig-zag line artifacts.
    """
    if TUNING_MODE != "random_search":
        return

    for model_name, history in fold_tuning_history.items():
        if not history:
            continue

        averaged_data = []
        for param_tuple, scores in history.items():
            param_dict = dict(param_tuple)
            avg_score = np.mean(scores) * 100
            param_dict['mean_accuracy'] = avg_score
            averaged_data.append(param_dict)

        df_params = pd.DataFrame(averaged_data)
        plt.figure(figsize=(9, 5))

        # -----------------------------------------------------------------
        # 1. LOGISTIC REGRESSION
        # -----------------------------------------------------------------
        if model_name == 'Logistic Regression':
            # Ensure C is numeric and sorted
            df_params['C'] = pd.to_numeric(df_params['C'], errors='coerce')

            for solver in df_params['solver'].unique():
                subset = df_params[df_params['solver'] == solver].sort_values(by='C')
                plt.plot(subset['C'], subset['mean_accuracy'], marker='o', label=f"solver={solver}")

            plt.xscale('log')  # Logarithmic scale for hyperparameter C (0.01 to 100)
            plt.xlabel("C Parameter (Inverse Regularization Strength - Log Scale)")
            plt.ylabel("Mean Accuracy Across All Folds (%)")
            plt.title(f"Hyperparameter Impact on Average Accuracy - {model_name}")
            plt.legend()
            plt.grid(True, which="both", ls="--", alpha=0.5)

        # -----------------------------------------------------------------
        # 2. SVM (RBF)
        # -----------------------------------------------------------------
        elif model_name == 'SVM (RBF)':
            # Filter numeric values for gamma vs string categorical options ('scale', 'auto')
            df_params['C'] = pd.to_numeric(df_params['C'], errors='coerce')

            for gamma_val in df_params['gamma'].unique():
                subset = df_params[df_params['gamma'] == gamma_val].sort_values(by='C')
                plt.plot(subset['C'], subset['mean_accuracy'], marker='o', label=f"gamma={gamma_val}")

            plt.xscale('log')  # Log scale for SVM C parameter
            plt.xlabel("C Parameter (Log Scale)")
            plt.ylabel("Mean Accuracy Across All Folds (%)")
            plt.title(f"Hyperparameter Impact on Average Accuracy - {model_name}")
            plt.legend()
            plt.grid(True, which="both", ls="--", alpha=0.5)

        # -----------------------------------------------------------------
        # 3. RANDOM FOREST
        # -----------------------------------------------------------------
        elif model_name == 'Random Forest':
            df_params['n_estimators'] = pd.to_numeric(df_params['n_estimators'], errors='coerce')
            # Handle None values in max_depth cleanly
            df_params['max_depth_str'] = df_params['max_depth'].apply(
                lambda x: 'None' if pd.isna(x) or x is None else str(x))

            for depth in df_params['max_depth_str'].unique():
                subset = df_params[df_params['max_depth_str'] == depth].sort_values(by='n_estimators')
                plt.plot(subset['n_estimators'], subset['mean_accuracy'], marker='s', label=f"max_depth={depth}")

            plt.xlabel("n_estimators (Number of Trees)")
            plt.ylabel("Mean Accuracy Across All Folds (%)")
            plt.title(f"Hyperparameter Impact on Average Accuracy - {model_name}")
            plt.legend()
            plt.grid(True, ls="--", alpha=0.5)

        # -----------------------------------------------------------------
        # 4. XGBOOST
        # -----------------------------------------------------------------
        elif model_name == 'XGBoost':
            df_params['learning_rate'] = pd.to_numeric(df_params['learning_rate'], errors='coerce')

            for depth in df_params['max_depth'].unique():
                subset = df_params[df_params['max_depth'] == depth].sort_values(by='learning_rate')
                plt.plot(subset['learning_rate'], subset['mean_accuracy'], marker='^', label=f"max_depth={depth}")

            plt.xlabel("Learning Rate")
            plt.ylabel("Mean Accuracy Across All Folds (%)")
            plt.title(f"Hyperparameter Impact on Average Accuracy - {model_name}")
            plt.legend()
            plt.grid(True, ls="--", alpha=0.5)

        # Save clean formatted plot to disk
        chart_path = current_dir / f"hyperparameter_impact_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()

def generate_overall_plots(df_summary, df_feat_summary):
    """
    Generates and saves final summary plots (Classifier comparison and Feature Frequency).
    """
    # 1. Classifier Overall Performance Plot
    plt.figure(figsize=(10, 5))
    models = df_summary['Classifier']
    accuracies = [float(val.split('%')[0]) for val in df_summary['Overall Accuracy']]

    plt.bar(models, accuracies, color='skyblue', edgecolor='black')
    plt.title("Overall Classifier Accuracy Comparison Across All Folds")
    plt.xlabel("Classifier")
    plt.ylabel("Mean Accuracy (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    chart_path = current_dir / "classifier_accuracy_comparison.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

    # 2. Top Selected Features Frequency Plot
    plt.figure(figsize=(10, 6))
    top_feats = df_feat_summary.head(15)
    plt.barh(top_feats['Feature'], top_feats['Selection_Frequency_Out_Of_10'], color='teal')
    plt.xlabel("Selection Frequency (Out of 10 Folds)")
    plt.title("Top Selected Acoustic Features (ANOVA)")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    feat_chart_path = current_dir / "top_features_frequency.png"
    plt.savefig(feat_chart_path, bbox_inches='tight')
    plt.close()


# =====================================================================
# MAIN RUNNER PROGRAM EXECUTION LOOP
# =====================================================================

def main():
    print("====================================================")
    print(f"HIGH-SPEED Multi-Classifier Nested Group K-Fold Loop ({RUN_MODE})")
    print(
        f"Feature Selection: {'ENABLED (Top ' + str(TOP_K_FEATURES) + ' ANOVA)' if ENABLE_FEATURE_SELECTION else 'DISABLED'}")
    print(f"Hyperparameter Tuning Mode: [{TUNING_MODE.upper()}]")
    print("====================================================")

    if RUN_MODE == "free_text":
        df_master = build_free_text_only_metadata()
    elif RUN_MODE == "mixed":
        df_master = build_mixed_metadata()
    elif RUN_MODE == "archive_only":
        df_master = build_archive_only_metadata()
    else:
        sys.exit(1)

    if df_master.empty:
        sys.exit(1)

    df_master_features = pre_extract_all_features_once(df_master)

    model_names = [
        'Gaussian NB', 'Logistic Regression', 'SVM (RBF)',
        'Random Forest', 'XGBoost', 'Kernel ELM'
    ]

    accents = ['chinese', 'indian', 'nigerian', 'usa']
    performance_tracker = {
        m: {
            'overall': [],
            'per_accent': {a: [] for a in accents},
            'y_true_all': [],
            'y_pred_all': []
        }
        for m in model_names
    }

    # Tracking tuning parameter history across folds for averaging
    fold_tuning_history = defaultdict(lambda: defaultdict(list))

    # Tracking selected features across all 10 folds
    all_selected_features_history = []

    fold_generator = run_nested_group_kfold_generator(df_master)

    for fold_idx, df_train_meta, df_val_meta, df_test_meta in fold_generator:
        print(f"\n" + "#" * 70)
        print(f" PROCESSING NESTED CROSS-VALIDATION FOLD SYSTEM: [FOLD {fold_idx}/10]")
        print(f"#" * 70)

        try:
            X_train, y_train, X_val, y_val, X_test, y_test, le, selected_feats = load_and_scale_fold_matrices(
                df_master_features, df_train_meta, df_val_meta, df_test_meta
            )

            # Record selected features for frequency analysis
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

                # Tune hyperparameters only for scikit-learn supported classifiers
                if model_name in ['Logistic Regression', 'SVM (RBF)', 'Random Forest', 'XGBoost']:
                    tuned_estimator, best_params = tune_model_hyperparameters(
                        model_name, X_train, y_train, fold_tuning_history, n_iter=60
                    )
                    if tuned_estimator is not None:
                        model_obj = tuned_estimator

                # Fit model
                model_obj.fit(X_train, y_train)

                # Predict on Test set
                y_pred = model_obj.predict(X_test)

                overall_acc = accuracy_score(y_test, y_pred)
                performance_tracker[model_name]['overall'].append(overall_acc)

                # Accumulate true & pred labels for global Confusion Matrix
                performance_tracker[model_name]['y_true_all'].extend(y_test)
                performance_tracker[model_name]['y_pred_all'].extend(y_pred)

                report = classification_report(
                    y_test, y_pred,
                    target_names=le.classes_,
                    output_dict=True,
                    zero_division=0
                )

                per_accent_str = []
                for acc_label in accents:
                    if acc_label in report:
                        acc_val = report[acc_label]['recall']
                        performance_tracker[model_name]['per_accent'][acc_label].append(acc_val)
                        per_accent_str.append(f"{acc_label[:3].upper()}: {acc_val * 100:.1f}%")

                print(f"   [{model_name:<19}] Overall: {overall_acc * 100:.2f}% | " + " | ".join(per_accent_str))

        except Exception as e:
            print(f"Error executing processing loop pipeline inside Fold {fold_idx}: {e}")
            traceback.print_exc()

    # =====================================================================
    # FINAL SUMMARY REPORT & BEST HYPERPARAMETERS EXPORT
    # =====================================================================
    print("\n" + "=" * 80)
    print("FINAL ACCENT CLASSIFICATION PERFORMANCE TABLE (NESTED GROUP K-FOLD)")
    print("=" * 80)

    summary_rows = []
    for model_name in model_names:
        tracker = performance_tracker[model_name]
        if not tracker['overall']:
            continue

        row = {'Classifier': model_name}
        mean_overall = np.mean(tracker['overall']) * 100
        std_overall = np.std(tracker['overall']) * 100
        row['Overall Accuracy'] = f"{mean_overall:.2f}% ± {std_overall:.2f}%"

        for acc_label in accents:
            acc_scores = tracker['per_accent'][acc_label]
            if acc_scores:
                row[f"{acc_label.capitalize()} Accuracy"] = f"{np.mean(acc_scores) * 100:.2f}%"
            else:
                row[f"{acc_label.capitalize()} Accuracy"] = "N/A"

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\n", df_summary.to_string(index=False))
    print("=" * 80)

    # Print Confusion Matrices for each classifier
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES PER CLASSIFIER (Aggregated over all folds)")
    print("=" * 80)
    for model_name in model_names:
        tracker = performance_tracker[model_name]
        if tracker['y_true_all']:
            cm = confusion_matrix(tracker['y_true_all'], tracker['y_pred_all'])
            print(f"\n--- {model_name} Confusion Matrix ---")
            print(f"Classes order: {le.classes_}")
            print(cm)
    print("=" * 80)

    # Print & export dedicated best hyperparameter report per classifier
    export_and_print_best_hyperparameters(fold_tuning_history)

    # =====================================================================
    # SELECTED FEATURES FREQUENCY ANALYSIS & CSV EXPORT
    # =====================================================================
    print("\n" + "=" * 80)
    print("TOP SELECTED ACOUSTIC FEATURES SUMMARY (Across All 10 Folds)")
    print("=" * 80)

    feat_counts = Counter(all_selected_features_history)
    df_feat_summary = pd.DataFrame(feat_counts.items(), columns=['Feature', 'Selection_Frequency_Out_Of_10'])
    df_feat_summary = df_feat_summary.sort_values(by='Selection_Frequency_Out_Of_10', ascending=False)

    print("\n", df_feat_summary.to_string(index=False))

    out_csv_path = current_dir / "top_selected_features_summary.csv"
    df_feat_summary.to_csv(out_csv_path, index=False)
    print(f"\n💾 Saved top selected features summary report to: {out_csv_path}")
    print("=" * 80)

    # Generate final plots
    generate_overall_plots(df_summary, df_feat_summary)
    generate_hyperparameter_averaged_plots(fold_tuning_history)


if __name__ == "__main__":
    main()