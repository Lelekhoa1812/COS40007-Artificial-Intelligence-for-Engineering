# A section import
import pandas as pd
import os
from sklearn.utils import shuffle

# B section import
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# C section import
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# D section import
from sklearn.feature_selection import SelectKBest, f_classif

# E section import
from sklearn.decomposition import PCA

# F and G section import
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC

# A. Studio Activity 1: Data preparation
# Define the folder containing the CSV files
folder_path = 'ampc'
# Step 1: Read and combine the CSV files
csv_files = ['w1.csv', 'w2.csv', 'w3.csv', 'w4.csv']
combined_df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in csv_files], ignore_index=True)
# Step 2: Save the combined data to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)
# Step 3: Shuffle the data
shuffled_df = shuffle(combined_df, random_state=42)
# Step 4: Save the shuffled data to a new CSV file
shuffled_df.to_csv('all_data.csv', index=False)
print("Data already prepared!") # Debug log print

# B. Studio Activity 2: Model Training
# Load the data
df = pd.read_csv('all_data.csv')
# Separate features and class
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column
# a. Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Train SVM model
clf = svm.SVC()
clf.fit(X_train, y_train)
# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy_train_test = accuracy_score(y_test, y_pred)
print(f"Accuracy with 70-30 train-test split: {accuracy_train_test:.4f}")
# b. 10-fold cross-validation
clf = svm.SVC()
scores = cross_val_score(clf, X, y, cv=10)
print("10-fold Cross-validation scores:", scores)
print(f"Mean accuracy with 10-fold cross-validation: {np.mean(scores):.4f}")
# Save the classification accuracy of the above 2 cases to a file (txt)
with open('svm_accuracies.txt', 'w') as f:
    f.write(f"Accuracy with 70-30 train-test split: {accuracy_train_test:.4f}\n")
    f.write(f"Mean accuracy with 10-fold cross-validation: {np.mean(scores):.4f}\n")
print("Accuracies have been saved to 'svm_accuracies.txt'") # Debug log print

# C. Studio Activity 3: Hyper parameter tuning
# Load the data
df = pd.read_csv('all_data.csv')
# Separate features and class
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
# Define parameter range for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
# Perform GridSearchCV
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)
# Print best parameters
print("Best parameters found by GridSearchCV:")
print(grid.best_params_)
# Get the best estimator
best_svm = grid.best_estimator_
# Train-test split accuracy with optimal parameters
y_pred = best_svm.predict(X_test)
accuracy_train_test = accuracy_score(y_test, y_pred)
print(f"Accuracy with 70-30 train-test split using optimal parameters: {accuracy_train_test:.4f}")
# 10-fold cross-validation with optimal parameters
scores = cross_val_score(best_svm, X_scaled, y, cv=10)
print("10-fold Cross-validation scores:", scores)
print(f"Mean accuracy with 10-fold cross-validation using optimal parameters: {np.mean(scores):.4f}")
# Save the accuracies to a file
with open('svm_accuracies_optimized.txt', 'w') as f:
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Accuracy with 70-30 train-test split using optimal parameters: {accuracy_train_test:.4f}\n")
    f.write(f"Mean accuracy with 10-fold cross-validation using optimal parameters: {np.mean(scores):.4f}\n")
print("Accuracies have been saved to 'svm_accuracies_optimized.txt'")

# D. Studio Activity 4: Feature Selection
# Step 1: Load the data
df = pd.read_csv('all_data.csv')
X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # The last column (target)
# Step 2: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 3: Select the 100 best features
selector = SelectKBest(f_classif, k=100)
X_kbest = selector.fit_transform(X_scaled, y)
# Step 4: Split the data into training and testing sets (70/30 split)
X_train_kbest, X_test_kbest, y_train, y_test = train_test_split(X_kbest, y, test_size=0.3, random_state=1)
# Step 5: Perform GridSearchCV with the 100 best features
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_kbest, y_train)
# a) Train-test split with hyperparameter tuning
y_pred_kbest = grid.best_estimator_.predict(X_test_kbest)
accuracy_kbest_train_test = accuracy_score(y_test, y_pred_kbest)
print(f"Accuracy with 70-30 train-test split (100 best features): {accuracy_kbest_train_test:.4f}")
# b) 10-fold cross-validation with hyperparameter tuning
scores_kbest = cross_val_score(grid.best_estimator_, X_kbest, y, cv=10)
accuracy_kbest_cv = scores_kbest.mean()
print(f"Mean accuracy with 10-fold cross-validation (100 best features): {accuracy_kbest_cv:.4f}")

# E. Studio Activity 5: Dimensionality Reduction with PCA
# Step 1: Apply PCA to reduce to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
# Step 2: Split the data into training and testing sets (70/30 split)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1)
# Step 3: Perform GridSearchCV with the PCA-transformed data
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid_pca = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)
grid_pca.fit(X_train_pca, y_train)
# a) Train-test split with hyperparameter tuning
y_pred_pca = grid_pca.best_estimator_.predict(X_test_pca)
accuracy_pca_train_test = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy with 70-30 train-test split (PCA 10 components): {accuracy_pca_train_test:.4f}")
# b) 10-fold cross-validation with hyperparameter tuning
scores_pca = cross_val_score(grid_pca.best_estimator_, X_pca, y, cv=10)
accuracy_pca_cv = scores_pca.mean()
print(f"Mean accuracy with 10-fold cross-validation (PCA 10 components): {accuracy_pca_cv:.4f}")
# Save the accuracies for Studio Activity 4 and 5 to a file
with open('svm_feature_selection_and_pca_accuracies.txt', 'w') as f:
    f.write(f"Accuracy with 70-30 train-test split (100 best features): {accuracy_kbest_train_test:.4f}\n")
    f.write(f"Mean accuracy with 10-fold cross-validation (100 best features): {accuracy_kbest_cv:.4f}\n")
    f.write(f"Accuracy with 70-30 train-test split (PCA 10 components): {accuracy_pca_train_test:.4f}\n")
    f.write(f"Mean accuracy with 10-fold cross-validation (PCA 10 components): {accuracy_pca_cv:.4f}\n")

# F. Studio Activity 6: Prepare a SVM summary table
# Load the data
df = pd.read_csv('all_data.csv')
X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # The last column (target)
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
# 1. SVM with Original Features
svm_original = SVC()
svm_original.fit(X_train, y_train)
original_split_acc = accuracy_score(y_test, svm_original.predict(X_test))
original_cv_acc = cross_val_score(svm_original, X_scaled, y, cv=10).mean()
# 2. SVM with Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
grid.fit(X_train, y_train)
svm_hyper = grid.best_estimator_
hyper_split_acc = accuracy_score(y_test, svm_hyper.predict(X_test))
hyper_cv_acc = cross_val_score(svm_hyper, X_scaled, y, cv=10).mean()
# 3. SVM with Feature Selection and Hyperparameter Tuning
selector = SelectKBest(f_classif, k=100)
X_kbest = selector.fit_transform(X_scaled, y)
X_train_kbest, X_test_kbest, y_train, y_test = train_test_split(X_kbest, y, test_size=0.3, random_state=1)
grid.fit(X_train_kbest, y_train)
svm_feature = grid.best_estimator_
feature_split_acc = accuracy_score(y_test, svm_feature.predict(X_test_kbest))
feature_cv_acc = cross_val_score(svm_feature, X_kbest, y, cv=10).mean()
# 4. SVM with PCA and Hyperparameter Tuning
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1)
grid.fit(X_train_pca, y_train)
svm_pca = grid.best_estimator_
pca_split_acc = accuracy_score(y_test, svm_pca.predict(X_test_pca))
pca_cv_acc = cross_val_score(svm_pca, X_pca, y, cv=10).mean()
# Function to prepare data for the SVM Models summary table
def generate_svm_summary_table(original_split_acc, original_cv_acc,
                               hyper_split_acc, hyper_cv_acc,
                               feature_split_acc, feature_cv_acc,
                               pca_split_acc, pca_cv_acc):
    # Creating the summary table
    summary_svm = pd.DataFrame({
        'SVM Model': [
            'SVM (Original features)',
            'SVM (With hyperparameter tuning)',
            'SVM (With feature selection and hyperparameter tuning)',
            'SVM (With PCA and hyperparameter tuning)'
        ],
        'Train-test split': [
            f'{original_split_acc:.2%}',
            f'{hyper_split_acc:.2%}',
            f'{feature_split_acc:.2%}',
            f'{pca_split_acc:.2%}'
        ],
        'Cross-validation': [
            f'{original_cv_acc:.2%}',
            f'{hyper_cv_acc:.2%}',
            f'{feature_cv_acc:.2%}',
            f'{pca_cv_acc:.2%}'
        ]
    })
    # Print the summary table
    print("Summary table for SVM Models:")
    print(summary_svm)
# Generate and print the SVM summary table
generate_svm_summary_table(
    original_split_acc=original_split_acc,
    original_cv_acc=original_cv_acc,
    hyper_split_acc=hyper_split_acc,
    hyper_cv_acc=hyper_cv_acc,
    feature_split_acc=feature_split_acc,
    feature_cv_acc=feature_cv_acc,
    pca_split_acc=pca_split_acc,
    pca_cv_acc=pca_cv_acc
)

# G. Studio Activity 7: Other classifiers
# Load the data
df = pd.read_csv('all_data.csv')
X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # The last column (target)
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
# 1. Train SGDClassifier
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train)
sgd_train_test_accuracy = accuracy_score(y_test, sgd.predict(X_test))
sgd_cv_accuracy = cross_val_score(sgd, X_scaled, y, cv=10).mean()
# 2. Train RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_train_test_accuracy = accuracy_score(y_test, rf.predict(X_test))
rf_cv_accuracy = cross_val_score(rf, X_scaled, y, cv=10).mean()
# 3. Train MLPClassifier
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)
mlp_train_test_accuracy = accuracy_score(y_test, mlp.predict(X_test))
mlp_cv_accuracy = cross_val_score(mlp, X_scaled, y, cv=10).mean()
# Print results
print(f"SGD: Train-test split accuracy: {sgd_train_test_accuracy:.4f}, Cross-validation accuracy: {sgd_cv_accuracy:.4f}")
print(f"RandomForest: Train-test split accuracy: {rf_train_test_accuracy:.4f}, Cross-validation accuracy: {rf_cv_accuracy:.4f}")
print(f"MLP: Train-test split accuracy: {mlp_train_test_accuracy:.4f}, Cross-validation accuracy: {mlp_cv_accuracy:.4f}")
# Print table
summary_all_models = pd.DataFrame({
    'Model': ['SVM', 'SGD', 'RandomForest', 'MLP'],
    'Train-test split': [
        '91.86%',  # Replace this with your SVM original model accuracy
        f'{sgd_train_test_accuracy:.2%}', 
        f'{rf_train_test_accuracy:.2%}', 
        f'{mlp_train_test_accuracy:.2%}'
    ],
    'Cross validation': [
        '91.79%',  # Replace this with your SVM original model accuracy
        f'{sgd_cv_accuracy:.2%}', 
        f'{rf_cv_accuracy:.2%}', 
        f'{mlp_cv_accuracy:.2%}'
    ]
})
print("Summary table for all Models:")
print(summary_all_models)
