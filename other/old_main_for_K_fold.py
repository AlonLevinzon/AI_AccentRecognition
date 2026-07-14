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
# FEATURE EXTRACTION ENGINE SUBROUTINE PER SPLIT FOLD
# =====================================================================

def extract_features_for_fold_split(df_split, split_name, fold_idx, cache_dir):
    """
    DYNAMIC EXTRACTION MANAGER: Writes temporary metadata files for the dynamic fold distribution slice,
    triggers 'import_vector_to_file' to load features, and cleans up the working directories.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Save a localized temporary metadata file for this specific split tracking segment
    temp_meta_csv = cache_dir / f"temp_meta_{split_name}_fold_{fold_idx}.csv"

    # Save the split rows with all standard final columns
    df_split[['filename', 'file_path', 'speaker_id', 'accent', 'text']].to_csv(temp_meta_csv, index=False,
                                                                               encoding='utf-8-sig')

    # 2. Trigger your original feature extraction subroutine script
    # FIXED: Parameter passed positionally to resolve your TypeError crash
    print(f"   ↳ Extracting audio acoustic vectors for {split_name} split ({len(df_split)} samples)...")
    import_vector_to_file(str(temp_meta_csv), None)

    # 3. Rename default output file destinations to isolate loops and prevent race condition overwrites
    default_output = Path("accent_features_dataset.csv")
    target_fold_csv = cache_dir / f"features_{split_name}_fold_{fold_idx}.csv"

    if default_output.exists():
        if target_fold_csv.exists():
            os.remove(target_fold_csv)
        os.rename(default_output, target_fold_csv)
    else:
        raise FileNotFoundError(
            f"Feature extraction engine failed to generate datasets for {split_name} fold {fold_idx}")

    # 4. Read features into active execution space memory and clean up temporary storage
    df_features = pd.read_csv(target_fold_csv)
    os.remove(temp_meta_csv)

    return df_features


def load_and_prepare_fold_matrices(df_train_meta, df_val_meta, df_test_meta, fold_idx, cache_dir):
    """
    Performs dynamically isolated feature extraction, cross-validation scaling,
    and target encoding exclusively inside the scope of the current Fold loop.
    """
    print(f"\n--- Initializing Feature Processing Pipeline for Fold {fold_idx} ---")

    # Step 1: Extract features independently for each dynamic slice to prevent leakages
    df_train = extract_features_for_fold_split(df_train_meta, 'train', fold_idx, cache_dir)
    df_val = extract_features_for_fold_split(df_val_meta, 'val', fold_idx, cache_dir)
    df_test = extract_features_for_fold_split(df_test_meta, 'test', fold_idx, cache_dir)

    # Step 2: Dynamic standard normalization scaling in RAM
    exclude_cols = ['filename', 'file_path', 'speaker_id', 'text', 'accent']
    features_to_scale = [col for col in df_train.columns if col not in exclude_cols]

    # Fit scaler strictly on the dynamic Train features slice only
    scaler = StandardScaler()

    # Extract values and transform splits safely
    X_train_scaled = scaler.fit_transform(df_train[features_to_scale])
    X_val_scaled = scaler.transform(df_val[features_to_scale])
    X_test_scaled = scaler.transform(df_test[features_to_scale])

    # Isolate independent numpy training matrices
    X_train = X_train_scaled.astype(float)
    X_val = X_val_scaled.astype(float)
    X_test = X_test_scaled.astype(float)

    # Step 3: Centralized Label Encoding safely bound to training targets
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['accent'])
    y_val = le.transform(df_val['accent'])
    y_test = le.transform(df_test['accent'])

    return X_train, y_train, X_val, y_val, X_test, y_test, le


# =====================================================================
# MAIN RUNNER PROGRAM EXECUTION LOOP
# =====================================================================

def main():
    print("=============================================")
    print("Multi-Classifier Nested Group K-Fold System")
    print("=============================================")

    # Select Mode: Toggle between "free_text" or "mixed" depending on dataset target configurations
    RUN_MODE = "mixed"

    current_dir = Path(__file__).resolve().parent
    cache_dir = current_dir / "temp_fold_features_cache"

    # 1. Initialize master matrices configurations
    if RUN_MODE == "free_text":
        df_master = build_free_text_only_metadata()
    else:
        df_master = build_mixed_metadata()

    if df_master.empty:
        print("🛑 Master dataframe initialization failed. Stopping execution.")
        sys.exit(1)

    # Dictionary to accumulate performance metrics across all loops for statistical tracking
    model_performance_history = {
        'Gaussian NB': [],
        'Logistic Regression': [],
        'SVM (RBF)': [],
        'Random Forest': [],
        'XGBoost': [],
        'Kernel ELM': []
    }

    # 2. Instantiate our Generator Engine from K_fold
    fold_generator = run_nested_group_kfold_generator(df_master)

    # 3. Dynamic Loop over every computed cross-validation fold structure
    for fold_idx, df_train_meta, df_val_meta, df_test_meta in fold_generator:
        print(f"\n" + "#" * 70)
        print(f" PROCESSING NESTED CROSS-VALIDATION FOLD SYSTEM: [FOLD {fold_idx}]")
        print(f"#" * 70)

        try:
            # Process, extract features, and scale dynamic fold matrices in RAM
            X_train, y_train, X_val, y_val, X_test, y_test, le = load_and_prepare_fold_matrices(
                df_train_meta, df_val_meta, df_test_meta, fold_idx, cache_dir
            )

            print(f"\nDynamic Fold Matrices established successfully:")
            print(f" - Train Matrix Shape: {X_train.shape} | Targets Split: {y_train.shape}")
            print(f" - Test Matrix Shape:  {X_test.shape}  | Targets Split: {y_test.shape}")

            # Instantiate a fresh dictionary of estimators per loop fold sequence
            models = {
                'Gaussian NB': get_gaussian_nb_classifier(),
                'Logistic Regression': get_logistic_regression_classifier(),
                'SVM (RBF)': get_svm_classifier(),
                'Random Forest': get_random_forest_classifier(),
                'XGBoost': get_xgboost_classifier(),
                'Kernel ELM': KernelELMClassifier()
            }

            # Train and benchmark performance across independent fold test subsets
            print(f"\nEvaluating Classifiers on Fold {fold_idx} Testing distribution...")
            for model_name, model_obj in models.items():
                print(f" -> Training {model_name}...")

                # Fit classifier on clean dynamic train matrix
                model_obj.fit(X_train, y_train)

                # Predict outcomes on completely unseen test data partition
                y_pred = model_obj.predict(X_test)

                # Output fold specific evaluation metric scores
                acc = accuracy_score(y_test, y_pred)
                print(f"   [{model_name}] Fold {fold_idx} Performance Accuracy: {acc * 100:.2f}%")

                # Append individual fold calculation to global stats history tracking engine
                model_performance_history[model_name].append(acc)

        except Exception as e:
            print(f"Error executing processing loop pipeline inside Fold {fold_idx}: {e}")
            traceback.print_exc()

        # COMMENT OUT OR REMOVE THE NEXT LINE TO EVALUATE ALL 10 FOLDS AND GENERATE FINAL MEANS!
        break

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
        else:
            print(f" -> {model_name:<20}: No completed fold histories logged.")


if __name__ == "__main__":
    main()