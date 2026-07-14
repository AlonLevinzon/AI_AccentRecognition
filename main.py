import os
import sys
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import classifiers from your existing project layout setup
from classifiers.GNB import get_gaussian_nb_classifier
from classifiers.KELM import KernelELMClassifier
from classifiers.logistic_regression_classifier import get_logistic_regression_classifier
from classifiers.random_forest_classifier import get_random_forest_classifier
from classifiers.svm_classifier import get_svm_classifier
from classifiers.XGBoost import get_xgboost_classifier

# Import your core connective engine components
from K_fold import build_free_text_only_metadata, build_mixed_metadata, run_nested_group_kfold_generator
from features.extract_features_to_files import import_vector_to_file

# =====================================================================
# GLOBAL CONFIGURATION FOR SPEED OPTIMIZATION
# =====================================================================
RUN_MODE = "free_text"  # Toggle between "free_text" or "mixed"

current_dir = Path(__file__).resolve().parent

# FIXED: Saved directly in your root project directory instead of a cache folder
EXTRACTED_FEATURES_MASTER_CSV = current_dir / f"master_extracted_features_{RUN_MODE}.csv"


def pre_extract_all_features_once(df_master):
    """
    ONE-TIME EXTRACTION: Runs the feature extraction loop exactly ONCE
    for the entire unified master list dataset, saving weeks of computational overhead.
    """
    if EXTRACTED_FEATURES_MASTER_CSV.exists():
        print(f"──► [CACHED DATA DETECTED]: Loading pre-extracted features from {EXTRACTED_FEATURES_MASTER_CSV}")
        return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)

    print(f"\n⚡⚡⚡ [FIRST RUN INITIALIZATION]: Extracting acoustic features for ALL {len(df_master)} files once...")
    print("This will take a while, but it will only run ONCE and cache the results permanently.")

    # 1. Create a transient master tracking metadata schema directly in project root
    temp_master_meta = current_dir / "temp_master_metadata_pipeline.csv"
    df_master.to_csv(temp_master_meta, index=False, encoding='utf-8-sig')

    # 2. Trigger your extraction engine for the entire unified scope (Using Positional arguments)
    import_vector_to_file(str(temp_master_meta), str(current_dir / "audio_by_accent_raw"))

    # 3. Cache the output permanently into the project root
    default_output = Path("accent_features_dataset.csv")
    if default_output.exists():
        if EXTRACTED_FEATURES_MASTER_CSV.exists():
            os.remove(EXTRACTED_FEATURES_MASTER_CSV)
        os.rename(default_output, EXTRACTED_FEATURES_MASTER_CSV)
        if temp_master_meta.exists():
            os.remove(temp_master_meta)
        print(f"──► [SUCCESS]: Cached unified dataset saved successfully at {EXTRACTED_FEATURES_MASTER_CSV}")
    else:
        raise FileNotFoundError(
            "The feature extraction subroutine failed to output results for caching processing loops.")

    return pd.read_csv(EXTRACTED_FEATURES_MASTER_CSV)


def filter_features_by_metadata_split(df_master_features, df_split_meta):
    """
    ULTRA FAST INDEX FILTER: Instead of scanning raw files, joins the pre-extracted
    features matrix with the specified fold metadata slice instantaneously via RAM.
    """
    # Merge on 'filename' because the extracted features CSV does not store 'file_path'
    df_merged = pd.merge(
        df_split_meta[['filename', 'file_path', 'speaker_id', 'text', 'accent']],
        df_master_features.drop(columns=['speaker_id', 'text', 'accent'], errors='ignore'),
        on='filename',
        how='inner'
    )

    return df_merged


def load_and_scale_fold_matrices(df_master_features, df_train_meta, df_val_meta, df_test_meta):
    """
    Performs memory-isolated feature splitting and cross-validation scaling
    without any disk or feature extraction bottlenecks.
    """
    # Instant RAM filtering using pre-computed master storage maps
    df_train = filter_features_by_metadata_split(df_master_features, df_train_meta)
    df_val = filter_features_by_metadata_split(df_master_features, df_val_meta)
    df_test = filter_features_by_metadata_split(df_master_features, df_test_meta)

    # Dynamic standard scaling normalization inside the fold execution boundaries
    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent']
    features_to_scale = [col for col in df_train.columns if col not in exclude_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[features_to_scale])
    X_val_scaled = scaler.transform(df_val[features_to_scale])
    X_test_scaled = scaler.transform(df_test[features_to_scale])

    X_train = X_train_scaled.astype(float)
    X_val = X_val_scaled.astype(float)
    X_test = X_test_scaled.astype(float)

    # Centralized multi-class categorical label target mappings encoding
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['accent'])
    y_val = le.transform(df_val['accent'])
    y_test = le.transform(df_test['accent'])

    return X_train, y_train, X_val, y_val, X_test, y_test, le


# =====================================================================
# MAIN RUNNER PROGRAM EXECUTION LOOP
# =====================================================================

def main():
    print("====================================================")
    print("HIGH-SPEED Multi-Classifier Nested Group K-Fold Loop")
    print("====================================================")

    # 1. Initialize or load the master layout tables
    if RUN_MODE == "free_text":
        df_master = build_free_text_only_metadata()
    else:
        df_master = build_mixed_metadata()

    if df_master.empty:
        print("🛑 Master dataframe initialization failed. Stopping execution.")
        sys.exit(1)

    # 2. Extract features exactly ONCE for all data points or load from disk cache
    df_master_features = pre_extract_all_features_once(df_master)

    # Performance metrics tracking history storage
    model_performance_history = {
        'Gaussian NB': [],
        'Logistic Regression': [],
        'SVM (RBF)': [],
        'Random Forest': [],
        'XGBoost': [],
        'Kernel ELM': []
    }

    # 3. Instantiate our Generator Engine from K_fold
    fold_generator = run_nested_group_kfold_generator(df_master)

    # 4. Fast Loop over cross-validation splits entirely inside memory (RAM)
    for fold_idx, df_train_meta, df_val_meta, df_test_meta in fold_generator:
        print(f"\n" + "#" * 70)
        print(f" PROCESSING NESTED CROSS-VALIDATION FOLD SYSTEM: [FOLD {fold_idx}/10]")
        print(f"#" * 70)

        try:
            # Load dynamic matrix chunks instantly from memory references
            X_train, y_train, X_val, y_val, X_test, y_test, le = load_and_scale_fold_matrices(
                df_master_features, df_train_meta, df_val_meta, df_test_meta
            )

            print(f" -> Matrix splits resolved instantly in RAM:")
            print(f"    Train Matrix Shape: {X_train.shape} | Test Matrix Shape: {X_test.shape}")

            # Instantiate fresh model estimators
            models = {
                'Gaussian NB': get_gaussian_nb_classifier(),
                'Logistic Regression': get_logistic_regression_classifier(),
                'SVM (RBF)': get_svm_classifier(),
                'Random Forest': get_random_forest_classifier(),
                'XGBoost': get_xgboost_classifier(),
                'Kernel ELM': KernelELMClassifier()
            }

            # Train and benchmark performance maps across fold distributions
            for model_name, model_obj in models.items():
                model_obj.fit(X_train, y_train)
                y_pred = model_obj.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                print(f"   [{model_name:<19}] Accuracy: {acc * 100:.2f}%")

                model_performance_history[model_name].append(acc)

        except Exception as e:
            print(f"Error executing processing loop pipeline inside Fold {fold_idx}: {e}")
            traceback.print_exc()

    # =====================================================================
    # FINAL CROSS-VALIDATION REPORT GENERATION PANEL
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL STRATIFIED NESTED CROSS-VALIDATION SUMMARY REPORT")
    print("=" * 70)

    for model_name, acc_list in model_performance_history.items():
        if acc_list:
            mean_acc = np.mean(acc_list) * 100
            std_acc = np.std(acc_list) * 100
            print(f" -> {model_name:<20}: {mean_acc:.2f}% ± {std_acc:.2f}% (Evaluated over {len(acc_list)} folds)")


if __name__ == "__main__":
    main()