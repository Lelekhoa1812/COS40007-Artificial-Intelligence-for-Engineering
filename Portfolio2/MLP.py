import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
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

# Initialize the MLP classifier with optimizations
mlp_model = MLPClassifier(
    hidden_layer_sizes=(30,),  # Reduce the number of neurons
    random_state=42, 
    max_iter=150,  # Further reduce max iterations
    early_stopping=True, 
    n_iter_no_change=5,  # Stop earlier if no improvement
    warm_start=True,  # Reuse the previous solution for faster convergence
    tol=1e-4  # Set a tolerance level to stop training early
)

# Train the model on the training data
mlp_model.fit(X_train, y_train)

# Predict on the test data
y_pred = mlp_model.predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Perform 3-fold cross-validation to speed up
cv_scores = cross_val_score(mlp_model, X_scaled, y, cv=3, scoring='accuracy')

# Print the accuracy results
print(f"Test Accuracy: {test_accuracy}")
print(f"Cross-Validation Accuracy: {cv_scores.mean()}")

# Save the accuracy results to a CSV file
accuracy_df = pd.DataFrame({
    'Metric': ['Test Accuracy', 'Cross-Validation Accuracy'],
    'Accuracy': [test_accuracy, cv_scores.mean()]
})
accuracy_df.to_csv('mlp_accuracy_results.csv', index=False)

print("Accuracy results have been saved to mlp_accuracy_results.csv")
