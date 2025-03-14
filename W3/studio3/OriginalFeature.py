import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
# Load the CSV files
w1 = pd.read_csv('w1.csv')
w2 = pd.read_csv('w2.csv')
w3 = pd.read_csv('w3.csv')
w4 = pd.read_csv('w4.csv')

# Combine the dataframes
combined_data = pd.concat([w1, w2, w3, w4], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_data.to_csv('all_data.csv', index=False)

df = pd.read_csv('all_data.csv')


# Separate features and class label
X = df.drop(columns=['class'])
y = df['class']

# Part a: Train-test split (70% train, 30% test) and measure accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = svm.SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

train_test_accuracy = accuracy_score(y_test, predictions)

print(f"Train-Test Split Accuracy: {train_test_accuracy:.2f}")

# Part b: 10-Fold Cross-Validation and measure accuracy
clf = svm.SVC()
cross_val_scores = cross_val_score(clf, X, y, cv=10)
cross_val_accuracy = cross_val_scores.mean()
print(f"10-Fold Cross-Validation Accuracy: {cross_val_accuracy:.2f}")

# Save the results in a dictionary or as a CSV
accuracy_results = {
    "Train-Test Split Accuracy": train_test_accuracy,
    "10-Fold Cross-Validation Accuracy": cross_val_accuracy
}

# Optionally, save the results to a CSV file
accuracy_df = pd.DataFrame([accuracy_results])
accuracy_df.to_csv('svm_accuracy_results.csv', index=False)

print("Accuracy results saved to 'svm_accuracy_results.csv'")
