import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("accent_features.csv")

X = df.drop(columns=['accent', 'speaker_id', 'filename'], errors='ignore')
y = df['accent']
# =====================================================================
# 1. DATA PREPARATION & ENCODING
# =====================================================================

# Convert categorical accent text labels (e.g., 'Arabic', 'Chinese') into numerical indices (0, 1, 2)
# Theoretical reason: XGBoost operates on mathematical loss functions and gradients,
# which require discrete numerical target indices for multiclass optimization.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training (80%) and testing (20%) subsets
# 'stratify=y_encoded' ensures class distribution balance across both splits to avoid evaluation bias.
# WARNING FOR ACCENT PROJECT: Standard sample-level split can lead to 'Speaker Leakage'.
# For the final evaluation, it is highly recommended to partition data based on 'speaker_id'
# (e.g., using GroupKFold) to guarantee the model evaluates on unseen voices.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =====================================================================
# 2. MODEL ARCHITECTURE & HYPERPARAMETERS
# =====================================================================

# Define the Ensemble Classifier using Gradient Boosted Decision Trees (GBDT)
xgb_model = XGBClassifier(
    # Ensemble Complexity:
    n_estimators=100,  # Total number of sequential Regression Trees (Weak Learners) to build.
    max_depth=4,  # Shallow trees constraint to limit individual variance and prevent overfitting.
    learning_rate=0.1,  # Shrinkage factor (eta) applied to each new tree's update to stabilize gradient descent.

    # Loss Function Formulation (Taylor 2nd Order Optimization):
    objective='multi:softprob',  # Multi-class objective function outputting predicted probability vectors per sample.
    eval_metric='mlogloss',
    # Multiclass Logarithmic Loss (Cross-Entropy) used to compute gradients (g_i) and Hessians (h_i).

    random_state=42
)

# =====================================================================
# 3. SEQUENTIAL TRAINING & INFERENCE
# =====================================================================

# Fit the boosting model onto the training data
# Mathematical process: Trees are built sequentially. Each split choice maximizes the theoretical 'Gain'
# metric based on the sum of first and second-order derivatives, heavily regularized by k-gamma and k-lambda parameters.
xgb_model.fit(X_train, y_train)

# Inference phase on unseen acoustic features
# The model aggregates continuous raw leaf outputs from all 100 trees, sums them up,
# applies a Softmax function, and maps the index back to the argmax class prediction.
y_pred = xgb_model.predict(X_test)

# =====================================================================
# 4. FEATURE IMPORTANCE EXTRACTION (Bonus for Presentation)
# =====================================================================

# Extract global Gain values across all trees to see which acoustic features contributed most to the splits
# Useful for the final report to prove which audio attributes (e.g., specific MFCC bands) dominate accent classification.
feature_importance = xgb_model.feature_importances_


#  (Precision, Recall, F1-Score)
print("--- Accuracy Score ---")
print(f"Total Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))