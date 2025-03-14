from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the data
df = pd.read_csv('all_data.csv')
X = df.drop(columns=['class'])
y = df['class']

# Define the model using RBF kernel
svc = SVC(kernel='rbf')

# Set up the parameter grid
param_grid = {
    'C': [1, 10],  # Regularization parameter
    'gamma': ['scale', 1]  # Kernel coefficient
}

# Configure GridSearchCV
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=10, scoring='accuracy', verbose=10)

# Perform grid search
grid_search.fit(X, y)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters found:", best_params)

# Reevaluate the model with optimal parameters using both train-test split and 10-fold CV
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
train_test_accuracy = accuracy_score(y_test, predictions)
print(f"Optimized Train-Test Split Accuracy: {train_test_accuracy:.2f}")

# 10-Fold Cross-Validation
cross_val_scores = cross_val_score(best_model, X, y, cv=10)
cross_val_accuracy = cross_val_scores.mean()
print(f"Optimized 10-Fold Cross-Validation Accuracy: {cross_val_accuracy:.2f}")

# Save the results in a dictionary or as a CSV
accuracy_results = {
    "Optimized Train-Test Split Accuracy": train_test_accuracy,
    "Optimized 10-Fold Cross-Validation Accuracy": cross_val_accuracy
}

# Optionally, save the results to a CSV file
accuracy_df = pd.DataFrame([accuracy_results])
accuracy_df.to_csv('optimized_svm_accuracy_results.csv', index=False)

print("Optimized accuracy results saved to 'optimized_svm_accuracy_results.csv'")
