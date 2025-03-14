import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('all_data.csv')

# Prepare data (assuming the last column is the target and the others are features)
X = df.drop(['Class'], axis=1)  # Replace 'Class' with your actual target column name if different
y = df['Class']

# Scale features (optional for Random Forest, but can help in certain cases)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier with optimizations
rf_model = RandomForestClassifier(
    n_estimators=50,  # Reduce number of trees to speed up
    max_depth=10,     # Limit the depth of the trees
    random_state=42,
    n_jobs=-1         # Use all available CPU cores
)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 5-fold cross-validation to speed up
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)

# Print the accuracy results
print(f"Test Accuracy: {test_accuracy}")
print(f"Cross-Validation Accuracy: {cv_scores.mean()}")

# Save the accuracy results to a CSV file
accuracy_df = pd.DataFrame({
    'Metric': ['Test Accuracy', 'Cross-Validation Accuracy'],
    'Accuracy': [test_accuracy, cv_scores.mean()]
})
accuracy_df.to_csv('rf_accuracy_results.csv', index=False)

print("Accuracy results have been saved to rf_accuracy_results.csv")
