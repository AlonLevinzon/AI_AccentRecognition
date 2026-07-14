import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from classifiers.GNB import get_gaussian_nb_classifier
from classifiers.KELM import KernelELMClassifier
from classifiers.logistic_regression_classifier import get_logistic_regression_classifier
from classifiers.random_forest_classifier import get_random_forest_classifier
from classifiers.svm_classifier import get_svm_classifier
from classifiers.XGBoost import get_xgboost_classifier
from sklearn.metrics import accuracy_score, classification_report
from features.extract_features_to_files import import_vector_to_file
from leakage_test import run_dataset_sanity_checks
import traceback


def load_and_prepare_features(BASE_DIR ):
    """
    Loads raw features, applies ColumnTransformer, and reconstructs the DataFrames
    ensuring 'filename', 'speaker_id', and 'text' are at the very beginning,
    and 'accent' is at the very end.
    """

    groups = ['train', 'validation', 'test']

    # 1. Create feature CSV files dynamically if they do not exist
    for group in groups:
        target_csv = os.path.join(BASE_DIR, f"{group}_features.csv")

        if not os.path.exists(target_csv):
            print(f"Feature file for '{group}' not found. Initializing extraction...")

            # MATCHING YOUR MANUAL MOVE: Looking for metadata directly in the root of BASE_DIR
            group_metadata_path = os.path.join(BASE_DIR, f"{group}_metadata.csv")
            group_audio_path = os.path.join(BASE_DIR, group)

            # Call the imported feature extraction engine
            import_vector_to_file(group_metadata_path, group_audio_path)

            # Rename the default script output to match the corresponding group dataset name
            if os.path.exists('accent_features_dataset.csv'):
                os.rename('accent_features_dataset.csv', target_csv)

    # Load the extracted RAW features from disk
    train_path = os.path.join(BASE_DIR, "train_features.csv")
    val_path = os.path.join(BASE_DIR, "validation_features.csv")
    test_path = os.path.join(BASE_DIR, "test_features.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print("Normalizing features dynamically in RAM...")

    # Identify numerical features to scale while preserving string metadata
    exclude_cols = ['filename', 'speaker_id', 'text', 'accent']
    features_to_scale = [col for col in df_train.columns if col not in exclude_cols]

    # Define the ColumnTransformer pipeline to prevent data leakage
    ct = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), features_to_scale)],
        remainder='passthrough'
    )

    # Fit transformer on training data and evaluate across validation/test splits exclusively
    scaled_train = ct.fit_transform(df_train)
    scaled_val = ct.transform(df_val)
    scaled_test = ct.transform(df_test)

    # Reconstruct columns internal ordering outputted by ColumnTransformer layout
    ct_output_order = features_to_scale + exclude_cols

    # Create temporary DataFrames matching ColumnTransformer's raw output sequence
    df_train_temp = pd.DataFrame(scaled_train, columns=ct_output_order)
    df_val_temp = pd.DataFrame(scaled_val, columns=ct_output_order)
    df_test_temp = pd.DataFrame(scaled_test, columns=ct_output_order)

    # Define strict logical sequence: Identifiers -> Scaled Numerical Features -> Categorical Target Label
    final_column_order = ['filename', 'speaker_id', 'text'] + features_to_scale + ['accent']

    # Reorder columns back to preferred analytical structure in memory
    df_train_scaled = df_train_temp[final_column_order]
    df_val_scaled = df_val_temp[final_column_order]
    df_test_scaled = df_test_temp[final_column_order]

    # Drop structural metadata and string descriptors to isolate model training matrix X
    drop_cols = ['filename', 'speaker_id', 'text', 'accent']

    X_train = df_train_scaled.drop(columns=drop_cols).astype(float)
    X_val = df_val_scaled.drop(columns=drop_cols).astype(float)
    X_test = df_test_scaled.drop(columns=drop_cols).astype(float)

    # Enforce centralized Label Encoding to prevent model crashes on multi-class setups
    le = LabelEncoder()
    y_train = le.fit_transform(df_train_scaled['accent'])
    y_val = le.transform(df_val_scaled['accent'])
    y_test = le.transform(df_test_scaled['accent'])

    return X_train, y_train, X_val, y_val, X_test, y_test, le


def main():
    print("=============================================")
    print("Initializing Multi-Classifier Accent System")
    print("=============================================")
    #
    # # 1. Run our rigorous leakage validation checks first!
    has_leakage = run_dataset_sanity_checks()

    if has_leakage:
        print("🛑 STOPPING INITIALIZATION: Please fix the dataset leakage bugs before training.")
        # Optional: You can uncomment sys.exit(1) if you want the code to forcefully stop on error
        sys.exit(1)
    #else:
        print("🚀 DATASET IS PURE: Proceeding to Model Training Loop...")

    # Load data splits, execute leak-free normalization, and get target label map
    #X_train, y_train, X_val, y_val, X_test, y_test, le = load_and_prepare_features( "official_audio_mixed")

    #all of the above but for free text only
    X_train, y_train, X_val, y_val, X_test, y_test, le = load_and_prepare_features("official_audio_free_text_only")

    print(f"\nData matrices initialized successfully:")
    print(f" - Train Shape:      {X_train.shape}")
    print(f" - Validation Shape: {X_val.shape}")
    print(f" - Test Shape:       {X_test.shape}")

    # Map containing custom instantiated classifiers
    models = {
        'Gaussian NB': get_gaussian_nb_classifier(),
        'Logistic Regression': get_logistic_regression_classifier(),
        'SVM (RBF)': get_svm_classifier(),
        'Random Forest': get_random_forest_classifier(),
        'XGBoost': get_xgboost_classifier(),
        'Kernel ELM': KernelELMClassifier()
    }

    # Iterate through classifiers dictionary, train models, and print evaluation benchmarks
    print("\nTraining and evaluating classifiers...")
    for model_name, model_obj in models.items():
        print(f"\n--- Running {model_name} ---")
        try:
            # Fit model on clean training data
            model_obj.fit(X_train, y_train)

            # Predict labels across test distribution matrix
            y_pred = model_obj.predict(X_test)

            # Compute general baseline accuracy metric score
            acc = accuracy_score(y_test, y_pred)
            print(f"{model_name} Test Accuracy: {acc * 100:.2f}%")

            # Print detailed evaluation breakdown metrics mapped back to string categories
            print(classification_report(y_test, y_pred, target_names=le.classes_))

        except Exception as e:
            print(f"Error executing validation pipeline for {model_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()