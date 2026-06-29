import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Prepare the data (Simulating your extracted audio features)
# In your real project, replace X and y with your real feature matrix and labels.
# X: Shape (n_samples, n_features) - e.g., MFCCs, Zero Crossing Rate, etc.
# y: Shape (n_samples,) - The accent labels (e.g., 'Hebrew', 'English', etc.)
X = np.random.randn(100, 13) 
y = np.random.choice(['Hebrew', 'English', 'Arabic'], size=100)

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Gaussian Naive Bayes classifier
gnb_model = GaussianNB()

# 4. Train the model on the training data
gnb_model.fit(X_train, y_train)

# 5. Predict the accent labels for the test data
y_pred = gnb_model.fit(X_train, y_train).predict(X_test)

# 6. Predict the probability distribution for each class (Optional baseline analysis)
probabilities = gnb_model.predict_proba(X_test)

# 7. Evaluate and print the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Print sample probabilities for the first 3 test samples
print("\nSample Class Probabilities (First 3 samples):")
print(probabilities[:3])