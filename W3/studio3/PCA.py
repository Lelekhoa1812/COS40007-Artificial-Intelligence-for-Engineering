from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your dataset
df = pd.read_csv('all_data.csv')
X = df.drop(columns=['class'])
y = df['class']

# Fit PCA on the entire dataset and transform it to get the first 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Assuming these were the best hyperparameters found previously
best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
svc = SVC(**best_params)

# a) Train-test split (70% train, 30% test) with hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=101)
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)
accuracy_train_test = accuracy_score(y_test, predictions)
print(f"Train-Test Split Accuracy with PCA: {accuracy_train_test:.2f}")

# b) 10-Fold Cross-Validation with hyperparameter tuning
accuracy_10_fold = cross_val_score(svc, X_pca, y, cv=10).mean()
print(f"10-Fold Cross-Validation Accuracy with PCA: {accuracy_10_fold:.2f}")

# Optionally, save the results
results_df = pd.DataFrame({
    'Method': ['Train-Test Split', '10-Fold Cross-Validation'],
    'Accuracy': [accuracy_train_test, accuracy_10_fold]
})
results_df.to_csv('svm_with_pca_results.csv', index=False)
print("Results saved to 'svm_with_pca_results.csv'")
