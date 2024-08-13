import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as mno
import itertools
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# # Import read file
# df = pd.read_csv("water_potability.csv")

# # A. Exploratory Data Analysis (EDA)
# # 2. Univariate Analysis
# print("Univariate Analysis")
# cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
# length = len(cols)
# cs = ["b","r","g","c","m","k","lime","c","orange"]
# fig = plt.figure(figsize=(7,7))
# for i, j, k in itertools.zip_longest(cols, range(length), cs):
#     plt.subplot(5,2,j+1)
#     ax = sns.histplot(df[i], color=k, kde=True)
#     ax.set_facecolor("w")
#     plt.axvline(df[i].mean(), linestyle="dashed", label="mean", color="k")
#     plt.legend(loc="best")
#     plt.title(i, color="navy")
#     plt.xlabel("")
# plt.tight_layout()
# plt.show()

# # 3. Summary Statistics
# print("Summary Statistics")
# print(df.describe().T)

# # 4. Multivariate Analysis
# print("Multivariate Analysis")
# sns.pairplot(df, diag_kind='kde', hue='Potability', corner = True)
# plt.savefig('multivariate_analysis.png', dpi=300)  # Save image as png figure as it's too big to view all (dpi is the resolution measurement)
# plt.show()

# # 5. Study Correlation
# # Check the Correlation
# print("Correlation ")
# print(df.corr())
# # Pairplot for checking the Correlation
# sns.pairplot(df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
#        'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']], kind = 'reg', corner = True)
# plt.savefig('correlation.png', dpi=300)  # Save image as png figure as it's too big to view all (dpi is the resolution measurement)
# plt.show()
# # Heatmap for checking the Correlation¶
# corr = abs(df.corr()) # correlation matrix
# lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
# mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap
# plt.figure(figsize = (7,7))
# sns.heatmap(lower_triangle, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
#             cbar= True, linewidths= 1, mask = mask)   # Da Heatmap
# plt.show()

# # B. Class labelling / Creating ground truth data
# # Function to classify 'Potability'
# def classify_potability(value):
#     if value == 0:
#         return 'Not potable'
#     elif value == 1:
#         return 'Potable'
#     else:
#         return 'Unknown'
# # Apply the function to create a new categorical column
# df['Potability_Class'] = df['Potability'].apply(classify_potability)
# # Save the modified DataFrame to a new CSV file
# df.to_csv('converted_water_potability.csv', index=False)
# # Plot the distribution of the classes in a bar chart
# class_distribution = df['Potability_Class'].value_counts().sort_index()
# class_distribution.plot(kind='bar', color='skyblue')
# plt.xlabel('Potability Class')
# plt.ylabel('Frequency')
# plt.title('Distribution of Potability Classes')
# plt.show()
# # Print the distribution for analysis
# print(class_distribution)

# # C. Feature Engineering
# converted_df = pd.read_csv('converted_water_potability.csv')
# # Define a function to categorize pH values
# def categorize_ph(pH):
#     if pH <= 2: # Very acidic 
#         return 1 
#     elif 2 < pH <= 4: # Acidic
#         return 2
#     elif 4 < pH <= 6: # Neutral
#         return 3
#     elif 6 < pH <= 8: # Basic
#         return 4
#     else: # Very basic
#         return 5 
# # Apply the function to create a new column 'pH_Class'
# converted_df['pH_Class'] = converted_df['ph'].apply(categorize_ph)
# # Update the modified dataframe
# converted_df.to_csv('converted_water_potability.csv', index=False)
# # Define new normalised dataset
# converted_df.to_csv('normalised_water_potability.csv', index=False)
# normalized_df = pd.read_csv('normalised_water_potability.csv')
# #  Select features to normalize
# features_to_normalize = ['Hardness', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Chloramines', 'Solids', 'Trihalomethanes']
# # Normalize the selected features
# scaler = MinMaxScaler()
# normalized_df[features_to_normalize] = scaler.fit_transform(normalized_df[features_to_normalize])
# normalized_df.to_csv('normalised_water_potability.csv', index=False)
# # Create composite features
# normalized_df['hardness_solids'] = normalized_df['Hardness'] + normalized_df['Solids']
# normalized_df['hardness_chloramines'] = normalized_df['Hardness'] + normalized_df['Chloramines']
# normalized_df['sulfate_organic_carbon'] = normalized_df['Sulfate'] + normalized_df['Organic_carbon']
# normalized_df['conductivity_organic_carbon'] = normalized_df['Conductivity'] + normalized_df['Organic_carbon']
# normalized_df['conductivity_trihalomethanes'] = normalized_df['Conductivity'] + normalized_df['Trihalomethanes']
# # Select the columns to save in the new CSV
# final_features = ['pH_Class'] + ['Potability'] + features_to_normalize + [
#     'hardness_solids', 
#     'hardness_chloramines', 
#     'sulfate_organic_carbon', 
#     'conductivity_organic_carbon', 
#     'conductivity_trihalomethanes'
# ]
# # # Save the DataFrame to a new CSV file
# normalized_df[final_features].to_csv('features_water_potability.csv', index=False)
# # Output the first few rows to check the new file
# print(normalized_df[final_features].head())

# # D. Feature Selection
# features_df = pd.read_csv('features_water_potability.csv')
# # Features to keep (and drop) based on EDA summary
# features_to_keep = [
#     'pH_Class',
#     'Potability',
#     'Hardness', 
#     'Sulfate', 
#     'Conductivity', 
#     'Organic_carbon',
#     'hardness_solids',  # Composite feature (Hardness + Solids)
#     'hardness_chloramines',  # Composite feature (Hardness + Chloramines)
#     'sulfate_organic_carbon',  # Composite feature (Sulfate + Organic_carbon)
#     'conductivity_organic_carbon',  # Composite feature (Conductivity + Organic_carbon)
#     'conductivity_trihalomethanes'  # Composite feature (Conductivity + Trihalomethanes)
# ]
# # Filter the dataframe to keep only the selected features
# df_selected = features_df[features_to_keep]
# # Save the new dataframe
# df_selected.to_csv('selected_features_water_potability.csv', index=False)

# # E. Create selected converted file
# converted_df = pd.read_csv('converted_water_potability.csv')
# # Select features and target variable
# selected_features = [
#     'Hardness', 'Sulfate', 'Conductivity', 'Organic_carbon', 
#     'Chloramines', 'Solids', 'Trihalomethanes', 'pH_Class', 'Potability'
# ]
# # Create a new dataframe with the selected features
# selected_df = converted_df[selected_features]
# # Save the new dataframe to a CSV file
# selected_df.to_csv('selected_converted_water_potability.csv', index=False)

# F. Training and decision tree Model development
def load_and_train_model(filename, feature_cols, target_col):
    # Load dataset
    df = pd.read_csv(filename)
    # Separate features and target variable
    X = df[feature_cols]  # Features
    y = df[target_col]  # Target variable
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy
# Define columns for each dataset
feature_cols_converted = [
    'Hardness', 'Sulfate', 'Conductivity', 'Organic_carbon', 
    'pH_Class'
]
target_col = 'Potability' 
# List of files
files = {
    "converted_water_potability.csv": feature_cols_converted,
    "normalised_water_potability.csv": feature_cols_converted,
    "features_water_potability.csv": feature_cols_converted,
    "selected_features_water_potability.csv": feature_cols_converted,
    "selected_converted_water_potability.csv": feature_cols_converted
}
# Evaluate models
for file, features in files.items():
    print(f"Evaluating model for {file}:")
    accuracy = load_and_train_model(file, features, target_col)
    print(f"Accuracy for {file}: {accuracy:.4f}")