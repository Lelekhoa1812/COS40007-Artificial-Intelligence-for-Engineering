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

# # Step 1: Data collection
# # Student ID ending with 1 (103844421)
# # => Column set 1: Right Shoulder (x,y,z)
# # => Column set 2: Left Shoulder (x,y,z)
# folder_path = 'ampc2'
# # Get csv files
# boning_df = pd.read_csv(os.path.join(folder_path, 'Boning.csv'))
# slicing_df = pd.read_csv(os.path.join(folder_path, 'Slicing.csv'))
# # Apply Class as 0 (Boning) and 1 (Slicing)
# boning_df['Class'] = 0
# slicing_df['Class'] = 1
# boning_df.to_csv('updated_Boning.csv', index=False)
# slicing_df.to_csv('updated_Slicing.csv', index=False)
# # Load the updated files
# updated_boning_df = pd.read_csv('updated_Boning.csv')
# updated_slicing_df = pd.read_csv('updated_Slicing.csv')
# # Filtrate column and combine the CSV files
# combined_df = pd.concat([updated_boning_df, updated_slicing_df], ignore_index=True)
# # Add the Frame column
# # Modified to increment per each row, to avoid duplicated frame id when coming 2 file
# combined_df['Frame'] = range(len(combined_df))
# keeping_features =  [
#     'Right Shoulder x', 
#     'Right Shoulder y', 
#     'Right Shoulder z', 
#     'Left Shoulder x', 
#     'Left Shoulder y', 
#     'Left Shoulder z', 
# ]
# # Save the combined dataset to CSV file
# combined_df[['Frame'] + keeping_features + ['Class']].to_csv('combined_data.csv', index=False)
# # Shuffle the data
# combined_df = shuffle(combined_df, random_state=42)
# combined_df.to_csv('combined_data.csv', index=False)

# # Step 2: Create composite columns
# # Load file
# combined_df = pd.read_csv('combined_data.csv')
# # Functions to calculate composite data points
# # Calculate root mean square, given there could be 2 or 3 entry values (a,b,c)
# # Define function to separate these 2 scenarios to calculate the root mean square
# def root_mean_square(a, b, c=None): 
#     if c is None: # Case 2 entries
#         return np.sqrt((a**2 + b**2) / 2)        # √(a^2+b^2)/2
#     else:         # Case 3 entries
#         return np.sqrt((a**2 + b**2 + c**2) / 3) # √(a^2+b^2+c^2)/3
# def roll(accelY, accelX, accelZ): # 180 * atan2(accelY, sqrt(accelX*accelX + accelZ*accelZ))/PI
#     return 180 * np.arctan2(accelY, np.sqrt(accelX**2 + accelZ**2)) / np.pi
# def pitch(accelX, accelY, accelZ):
#     return 180 * np.arctan2(accelX, np.sqrt(accelY**2 + accelZ**2)) / np.pi
# # Call functions to calculate composite data points for Column Set 1 (Right Shoulder)
# combined_df['Right Shoulder RMS xy'] = root_mean_square(combined_df['Right Shoulder x'], combined_df['Right Shoulder y'])
# combined_df['Right Shoulder RMS yz'] = root_mean_square(combined_df['Right Shoulder y'], combined_df['Right Shoulder z'])
# combined_df['Right Shoulder RMS zx'] = root_mean_square(combined_df['Right Shoulder z'], combined_df['Right Shoulder x'])
# combined_df['Right Shoulder RMS xyz'] = root_mean_square(combined_df['Right Shoulder x'], combined_df['Right Shoulder y'], combined_df['Right Shoulder z'])
# combined_df['Right Shoulder Roll'] = roll(combined_df['Right Shoulder y'], combined_df['Right Shoulder x'], combined_df['Right Shoulder z'])
# combined_df['Right Shoulder Pitch'] = pitch(combined_df['Right Shoulder x'], combined_df['Right Shoulder y'], combined_df['Right Shoulder z'])
# # Call functions to calculate composite data points for Column Set 2 (Left Shoulder)
# combined_df['Left Shoulder RMS xy'] = root_mean_square(combined_df['Left Shoulder x'], combined_df['Left Shoulder y'])
# combined_df['Left Shoulder RMS yz'] = root_mean_square(combined_df['Left Shoulder y'], combined_df['Left Shoulder z'])
# combined_df['Left Shoulder RMS zx'] = root_mean_square(combined_df['Left Shoulder z'], combined_df['Left Shoulder x'])
# combined_df['Left Shoulder RMS xyz'] = root_mean_square(combined_df['Left Shoulder x'], combined_df['Left Shoulder y'], combined_df['Left Shoulder z'])
# combined_df['Left Shoulder Roll'] = roll(combined_df['Left Shoulder y'], combined_df['Left Shoulder x'], combined_df['Left Shoulder z'])
# combined_df['Left Shoulder Pitch'] = pitch(combined_df['Left Shoulder x'], combined_df['Left Shoulder y'], combined_df['Left Shoulder z'])
# # Restructure columns order
# final_columns = [
#     'Frame',
#     'Right Shoulder x', 'Right Shoulder y', 'Right Shoulder z',
#     'Left Shoulder x', 'Left Shoulder y', 'Left Shoulder z',
#     'Right Shoulder RMS xy', 'Right Shoulder RMS yz', 'Right Shoulder RMS zx', 'Right Shoulder RMS xyz', 'Right Shoulder Roll', 'Right Shoulder Pitch',
#     'Left Shoulder RMS xy', 'Left Shoulder RMS yz', 'Left Shoulder RMS zx', 'Left Shoulder RMS xyz', 'Left Shoulder Roll', 'Left Shoulder Pitch',
#     'Class'
# ]
# # Save the updated composite dataset to CSV file
# combined_df[final_columns].to_csv('composite_data.csv', index=False)

# # Step 3: Data pre-processing and Feature computation
# # Load file
# composite_df = pd.read_csv('composite_data.csv')
# # Set columns to compute statistics (2-19)
# columns_to_compute = composite_df.columns[1:19]
# # Function to compute features for a single window
# def compute_statistical_features(window):
#     features = {} # Initialize an empty array to store computed features
#     features['Frame'] = window['Frame'].iloc[0] # Include Frame as a feature
#     for col in columns_to_compute:
#         data = window[col].values # Extract the values of current column for the window
#         # Compute statistical features
#         features[f'{col}_mean'] = np.mean(data)             # Mean
#         features[f'{col}_std'] = np.std(data)               # Standard deviation
#         features[f'{col}_min'] = np.min(data)               # Min value
#         features[f'{col}_max'] = np.max(data)               # Max value
#         features[f'{col}_auc'] = np.trapz(data)             # Area under the curve (AUC) 
#         features[f'{col}_peaks'] = len(find_peaks(data)[0]) # Number of peaks
#     features['Class'] = window['Class'].iloc[0] # Include Class as a feature
#     return features
# # Create an empty list to hold the results
# statistical_features = []
# # Process the DataFrame in windows of 60 frames
# window_size = 60
# for i in range(0, len(composite_df), window_size):
#     # Extract the window of data
#     window = composite_df.iloc[i:i+window_size]
#     # Compute the features for this window
#     features = compute_statistical_features(window)
#     # Append the result to the list
#     statistical_features.append(features)
# # Convert the list of dictionaries to a DataFrame
# stat_features_df = pd.DataFrame(statistical_features)
# # Save the resulting DataFrame to a CSV file
# stat_features_df.to_csv('statistical_features.csv', index=False)

# Step 4: Training
# Load file
stat_features_df = pd.read_csv('statistical_features.csv')
# Separate features and target
X = stat_features_df.drop(columns=['Class']) # Drop non-featured columns
y = stat_features_df['Class']                # Target column

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
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=10)
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

