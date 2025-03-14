import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
df = pd.read_csv('minute_features.csv')

# Prepare data
X = df.drop(['Class', 'Frame Start', 'Frame End'], axis=1)  # Drop non-feature columns
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection: Select the top 10 best features
selector = SelectKBest(f_classif, k=10)
X_best_features = selector.fit_transform(X_scaled, y)

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_best_features, y, test_size=0.3, random_state=42)

# Define a parameter grid to search for best parameters for SVM
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['linear', 'rbf']  # Specifies the kernel type to be used in the algorithm
}

# Initialize SVM classifier
svm_model = SVC()

# Setup GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=1)

# Perform grid search to find the best parameters
grid_search.fit(X_train, y_train)

# Best model after grid search
best_model = grid_search.best_estimator_

# Predict on test data using the best model
y_pred = best_model.predict(X_test)

# Evaluation
test_accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)
classification_df = pd.DataFrame(classification_rep).transpose()

# Print outputs
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
print("Test Accuracy:", test_accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prepare data for saving
output_df = pd.DataFrame({
    'Metric': ['Best Parameters', 'Best Cross-Validation Accuracy', 'Test Accuracy'],
    'Value': [str(grid_search.best_params_), grid_search.best_score_, test_accuracy]
})

# Append classification report details
classification_df['Metric'] = classification_df.index
classification_df = classification_df.reset_index(drop=True)

# Concatenate all data
final_output_df = pd.concat([output_df, classification_df], ignore_index=True)

# Save all output to a CSV file
final_output_df.to_csv('svm_full_output_with_features.csv', index=False)

# Additionally, save the selected feature indices and names
selected_features = pd.DataFrame({
    'Selected Feature Index': selector.get_support(indices=True),
    'Selected Feature Name': X.columns[selector.get_support(indices=True)]
})
selected_features.to_csv('selected_features.csv', index=False)

print("All outputs and selected features have been saved to svm_full_output_with_features.csv and selected_features.csv")
