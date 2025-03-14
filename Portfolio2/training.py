import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
df = pd.read_csv('minute_features.csv')

# Prepare data
X = df.drop(['Class', 'Frame Start', 'Frame End'], axis=1)  # Drop non-feature columns
y = df['Class']

# Feature selection to reduce dimensionality
selector = SelectKBest(f_classif, k=10)  # Select the top 10 features
X_reduced = selector.fit_transform(X, y)

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Initialize SVM classifier with a simpler kernel
svm_model = SVC(kernel='linear')  # Linear is faster than RBF for large datasets

# Perform cross-validation (Reduced to 5 folds for speed)
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)

# Train the classifier on the training data
svm_model.fit(X_train, y_train)

# Predict on the test data
y_pred = svm_model.predict(X_test)

# Evaluation
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save results
results = {
    "Cross-Validation Scores": cv_scores.tolist(),
    "Mean CV Accuracy": cv_scores.mean(),
    "Test Accuracy": accuracy_score(y_test, y_pred),
}
results_df = pd.DataFrame([results])
results_df.to_csv('svm_results.csv', index=False)
