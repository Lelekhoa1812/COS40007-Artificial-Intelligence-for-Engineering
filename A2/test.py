# Step 1 import 
import pandas as pd
import os
from sklearn.utils import shuffle

# Step 2 import
import numpy as np

# Step 3 import
from scipy.signal import find_peaks

# Step 4 import
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 4: Training
# Load file
stat_features_df = pd.read_csv('statistical_features.csv')
# Separate features and target
X = stat_features_df.drop(columns=['Class','Frame']) # Drop non-featured columns
y = stat_features_df['Class']

# 1) Train-Test Split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2) Train SVM with default settings and perform 10-fold cross-validation
svm = SVC()
svm.fit(X_train, y_train)
# Evaluate on test data
test_accuracy_2 = svm.score(X_test, y_test)
# Cross-validation (only on training data)
cv_scores = cross_val_score(svm, X_train, y_train, cv=10)
cv_accuracy_2 = cv_scores.mean()
# Store results
results = {
    'Model': 'SVM (Original features)',
    'Train-test split': f'{test_accuracy_2:.2%}',
    'Cross validation': f'{cv_accuracy_2:.2%}'
}

# 3) Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
# Best parameters
best_params = grid.best_params_
# Evaluate on test data with the best parameters
test_accuracy_3 = grid.score(X_test, y_test)
# Cross-validation (only on training data)
cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=10)
cv_accuracy_3 = cv_scores.mean()
# Store results
results_hyper = {
    'Model': 'SVM (With hyperparameter tuning)',
    'Train-test split': f'{test_accuracy_3:.2%}',
    'Cross validation': f'{cv_accuracy_3:.2%}'
}

# 4) Feature selection with SelectKBest
selector = SelectKBest(f_classif, k=10)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)
# Train and evaluate with hyperparameter tuning
grid.fit(X_train_kbest, y_train)
test_accuracy_4 = grid.score(X_test_kbest, y_test)
# Cross-validation (only on training data)
cv_scores = cross_val_score(grid.best_estimator_, X_train_kbest, y_train, cv=10)
cv_accuracy_4 = cv_scores.mean()
# Store results
results_feature_selection = {
    'Model': 'SVM (With feature selection and hyperparameter tuning)',
    'Train-test split': f'{test_accuracy_4:.2%}',
    'Cross validation': f'{cv_accuracy_4:.2%}'
}

# 5) PCA for dimensionality reduction
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# Train and evaluate with hyperparameter tuning
grid.fit(X_train_pca, y_train)
test_accuracy_5 = grid.score(X_test_pca, y_test)
# Cross-validation (only on training data)
cv_scores = cross_val_score(grid.best_estimator_, X_train_pca, y_train, cv=10)
cv_accuracy_5 = cv_scores.mean()
# Store results
results_pca = {
    'Model': 'SVM (With PCA and hyperparameter tuning)',
    'Train-test split': f'{test_accuracy_5:.2%}',
    'Cross validation': f'{cv_accuracy_5:.2%}'
}

# 6) Summary table for SVM Models
summary_svm = pd.DataFrame([results, results_hyper, results_feature_selection, results_pca])
print("Summary table for SVM Models")
print(summary_svm)
print('-----------------------------------------------------------------------------') # Split 2 tables

# 7) Train SGD, RandomForest, and MLP Classifiers
# Train SGDClassifier
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train)
sgd_test_accuracy = sgd.score(X_test, y_test)
sgd_cv_accuracy = cross_val_score(sgd, X, y, cv=10).mean()
# Train RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_test_accuracy = rf.score(X_test, y_test)
rf_cv_accuracy = cross_val_score(rf, X, y, cv=10).mean()
# Train MLPClassifier
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)
mlp_test_accuracy = mlp.score(X_test, y_test)
mlp_cv_accuracy = cross_val_score(mlp, X, y, cv=10).mean()

# 8) Summary table for all Models
# Create a summary table for all models
summary_all_models = pd.DataFrame({
    'Model': ['SVM', 'SGD', 'RandomForest', 'MLP'],
    'Train-test split': [
        results['Train-test split'], 
        f'{sgd_test_accuracy:.2%}', 
        f'{rf_test_accuracy:.2%}', 
        f'{mlp_test_accuracy:.2%}'
    ],
    'Cross validation': [
        results['Cross validation'], 
        f'{sgd_cv_accuracy:.2%}', 
        f'{rf_cv_accuracy:.2%}', 
        f'{mlp_cv_accuracy:.2%}'
    ]
})
print("Summary table for all Models")
print(summary_all_models)

