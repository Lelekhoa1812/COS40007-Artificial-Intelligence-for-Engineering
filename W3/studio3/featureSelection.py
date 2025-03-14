from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your dataset
df = pd.read_csv('all_data.csv')
X = df.drop(columns=['class'])
y = df['class']

# Select the top 100 features using K-Best
selector = SelectKBest(score_func=f_classif, k=100)
X_new = selector.fit_transform(X, y)

# Assuming these were the best hyperparameters found previously
best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
svc = SVC(**best_params)

# a) Train-test split (70% train, 30% test) with hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=101)
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)
accuracy_train_test = accuracy_score(y_test, predictions)
print(f"Train-Test Split Accuracy with 100 Best Features: {accuracy_train_test:.2f}")

# b) 10-Fold Cross-Validation with hyperparameter tuning
accuracy_10_fold = cross_val_score(svc, X_new, y, cv=10).mean()
print(f"10-Fold Cross-Validation Accuracy with 100 Best Features: {accuracy_10_fold:.2f}")

# Optionally, save the results
results_df = pd.DataFrame({
    'Method': ['Train-Test Split', '10-Fold Cross-Validation'],
    'Accuracy': [accuracy_train_test, accuracy_10_fold]
})
results_df.to_csv('svm_with_100_best_features_results.csv', index=False)
print("Results saved to 'svm_with_100_best_features_results.csv'")
