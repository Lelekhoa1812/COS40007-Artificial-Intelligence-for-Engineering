# STEP 1: Data Preparation
# A. Preparing Data
import pandas as pd
from sklearn.utils import shuffle
# Load the dataset
vegemite_df = pd.read_csv('vegemite.csv')
# Shuffle the dataset
vegemite_df = shuffle(vegemite_df, random_state=42)

# 2) Take out 1000 samples ensuring near equal distribution across classes
from sklearn.model_selection import train_test_split
# Stratified split to ensure near-equal distribution of classes
_, test_data = train_test_split(vegemite_df, test_size=1000, stratify=vegemite_df['Class'], random_state=42)
# Use remaining data for training
train_data = vegemite_df.drop(test_data.index)

# B. Constructing Data
# 1) Remove constant value columns
train_data = train_data.loc[:, (train_data != train_data.iloc[0]).any()]
test_data = test_data.loc[:, (test_data != test_data.iloc[0]).any()] 

# 2) Convert columns with few integer values to categorical features in train and test dataset
for col in train_data.select_dtypes(include='int64').columns:
    train_data[col] = train_data[col].astype('category')
    if col in test_data.columns:
        test_data[col] = test_data[col].astype('category')

# 3) Check class balance
from imblearn.over_sampling import SMOTE
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 4) Composite feature exploration 
# import seaborn as sns
# import matplotlib.pyplot as plt
# # Compute the correlation matrix
# correlation_matrix = train_data.corr()
# # Save correlation matrix created to csv file for better understanding of the data
# correlation_matrix.to_csv('correlation_matrix.csv', index=False)
# # Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(8, 8))
# sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
# plt.title('Correlation Matrix Heatmap')
# plt.show()
# Create a composite feature with the strongest correlation feature, found to be 'FFTE Production solids PV' and 'FFTE Discharge density'
train_data['Composite_Feature'] = train_data['FFTE Production solids PV'] * train_data['FFTE Discharge density']
if 'FFTE Production solids PV' in test_data.columns and 'FFTE Discharge density' in test_data.columns:
    test_data['Composite_Feature'] = test_data['FFTE Production solids PV'] * test_data['FFTE Discharge density']
# After SMOTE in part 3, add the composite feature back to the resampled dataset
X_resampled['Composite_Feature'] = X_resampled['FFTE Production solids PV'] * X_resampled['FFTE Discharge density']

# 5) Print final number of features
# Drop any non-feature columns before printing
final_features = train_data.drop(columns=['Class'])
print(f"Final number of features: {final_features.shape[1]}")

print("-------------------------------------------------------------------") # Splitter

# STEP 2: Feature selection, Model Training, and Evaluation
# 6) Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
# Ensure that both training and test data have the same columns
common_columns = train_data.columns.intersection(test_data.columns).drop('Class')
X_resampled = X_resampled[common_columns]
X_test = test_data[common_columns]
y_test = test_data['Class']
# Feature selection
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_resampled, y_resampled)
X_test_selected = selector.transform(X_test) 

# 7) Train multiple ML models
# Prepare test data (make sure it aligns with selected features)
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']
X_test_selected = selector.transform(X_test)  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Scale the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'SGD': SGDClassifier(),
    'MLP': MLPClassifier(max_iter=1000)
}

# 8) Evaluate each model with classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
results = []
for name, model in models.items():
    model.fit(X_selected, y_resampled)
    y_pred = model.predict(X_test_selected)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Collect accuracy for comparison
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': accuracy})

# 9) Compare all the models and generate comparison table
results_df = pd.DataFrame(results)
print("Comparison of models:")
print(results_df)

# 10) Select the best model
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# 11) Save the selected model
import joblib
joblib.dump(best_model, 'best_model.pkl')
print("Model saved as 'best_model.pkl'")

print("-------------------------------------------------------------------") # Splitter

# STEP 3: ML to AI
# 12) Load the 1000 rows (test_data) that were set aside at the beginning
test_data = pd.read_csv('vegemite_test.csv')

# 13) Load the best model
best_model = joblib.load('best_model.pkl')

# 14) Prepare test data in the format of your training feature set
test_data['Composite_Feature'] = test_data['FFTE Production solids PV'] * test_data['FFTE Discharge density']
X_test = test_data[common_columns]  # Ensure the features match
# Apply the same feature selection
X_test_selected = selector.transform(X_test)
# Refit the scaler on the selected features only
scaler_selected = StandardScaler()
X_selected_scaled = scaler_selected.fit_transform(X_selected)  # Scale selected features for training
X_test_scaled = scaler_selected.transform(X_test_selected)  # Scale selected features for testing

# 15) Predict the class using the loaded model
y_pred = best_model.predict(X_test_scaled)

# 16) Measure the performance of your best model on the unseen data
print("Performance of the Best Model on Unseen Data:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 17) Measure the performance of other models on the 1000 data points
print("\nPerformance of Other Models on Unseen Data:")
for name, model in models.items():
    # Ensure the model is trained on the same reduced feature set (20 features)
    model.fit(X_selected_scaled, y_resampled)  # Train the model on the scaled training data
    y_pred_other = model.predict(X_test_scaled)  # Predict on the 1000 test data points
    accuracy = accuracy_score(y_test, y_pred_other)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred_other))
    print(f"Accuracy: {accuracy:.2%}\n")

print("-------------------------------------------------------------------") # Splitter

# STEP 4: Develop Rules from ML Model
from sklearn.tree import export_text
# Filter only columns that end with 'SP'
sp_columns = [col for col in train_data.columns if col.endswith('SP')]

#  Train the decision tree model 
X_sp = train_data[sp_columns]
y_sp = train_data['Class']
# Train a decision tree model using only SP features
dt_sp = DecisionTreeClassifier()
dt_sp.fit(X_sp, y_sp)

# Print the decision tree as text (rules)
tree_rules = export_text(dt_sp, feature_names=sp_columns)       
print("Decision Tree Rules Based on SP Features:")

# Save the rules to a text file
with open('rule.txt', 'w') as f:    
    f.write(tree_rules)
print("Rules saved to rule.txt")
