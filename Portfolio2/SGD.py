import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('all_data.csv')

# Prepare data (assuming the last column is the target and the others are features)
X = df.drop(['Class'], axis=1)  # Replace 'Class' with your actual target column name if different
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the SGD classifier
sgd_model = SGDClassifier(random_state=42)

# Train the model on the training data
sgd_model.fit(X_train, y_train)

# Predict on the test data
y_pred = sgd_model.predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(sgd_model, X_scaled, y, cv=10, scoring='accuracy')

# Print the accuracy results
print(f"Test Accuracy: {test_accuracy}")
print(f"Cross-Validation Accuracy: {cv_scores.mean()}")

# Save the accuracy results to a CSV file
accuracy_df = pd.DataFrame({
    'Metric': ['Test Accuracy', 'Cross-Validation Accuracy'],
    'Accuracy': [test_accuracy, cv_scores.mean()]
})
accuracy_df.to_csv('sgd_accuracy_results.csv', index=False)

print("Accuracy results have been saved to sgd_accuracy_results.csv")
