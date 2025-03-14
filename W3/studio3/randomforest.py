from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
df = pd.read_csv('all_data.csv')
X = df.drop(columns=['class'])  # Assume 'class' is the target column
y = df['class']

# Initialize the RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,  # Number of trees in the forest
                                       criterion='gini',  # Measure the quality of a split
                                       max_depth=None,    # Maximum depth of the tree
                                       random_state=42)   # For reproducibility

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)
accuracy_train_test = accuracy_score(y_test, predictions)
print(f"Train-Test Split Accuracy: {accuracy_train_test:.2f}")

# Cross-validation
accuracy_cross_val = cross_val_score(random_forest, X, y, cv=10).mean()  # 10-fold cross-validation
print(f"10-Fold Cross-Validation Accuracy: {accuracy_cross_val:.2f}")

# Save the results in a dictionary or as a CSV
accuracy_results = {
    "Train-Test Split Accuracy": accuracy_train_test,
    "10-Fold Cross-Validation Accuracy": accuracy_cross_val
}

# Optionally, save the results to a CSV file
accuracy_df = pd.DataFrame([accuracy_results])
accuracy_df.to_csv('random_forest_accuracy_results.csv', index=False)

print("Accuracy results saved to 'random_forest_accuracy_results.csv'")
