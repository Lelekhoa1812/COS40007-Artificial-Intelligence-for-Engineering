## IMPORT DATA FROM GOOGLE DRIVE
# from google.colab import drive
import pandas as pd                            # general usage
from sklearn.preprocessing import MinMaxScaler # data preprocessing
from sklearn.preprocessing import LabelEncoder # data preprocessing
from sklearn.decomposition import PCA          # data preprocessing
import matplotlib.pyplot as plt                # features engineer
import seaborn as sns                          # features engineer

# # Mount Google Drive
# drive.mount('/content/drive')
# # Load the dataset from Google Drive
# file_path = '/content/drive/My Drive/LTE_KPI.csv'
# data = pd.read_csv(file_path)

# Local
data = pd.read_csv('LTE_KPI.csv')

## HANDLE MISSING VALUE
# removeDV.py:
# Identify columns containing "#DIV/0"
div0_columns = [col for col in data.columns if data[col].astype(str).str.contains("#DIV/0").any()]
# Remove the identified columns
cleaned_data = data.drop(columns=div0_columns)
# Save the cleaned data to a new CSV file
# cleaned_file_path = 'cleanedDV.csv'                 # Un-comment this to actually save CSV file
# cleaned_data.to_csv(cleaned_file_path, index=False) # Un-comment this to actually save CSV file
# print(f"Cleaned data saved to {cleaned_file_path}") # Un-comment this to actually save CSV file

# removeMissingvalue.py:
# Remove columns with all missing values
data_cleaned = cleaned_data.dropna(axis=1, how='all')
# Convert the 'Date' column to datetime format
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])
# Save the cleaned data to a new CSV file
# data_cleaned.to_csv('Removed_Missing_Values.csv', index=False)                # Un-comment this to actually save CSV file
# print("Cleaned data has been saved. Here's the info of the cleaned dataset:") # Un-comment this to actually save CSV file
print(data_cleaned.info())

# removeRow_missingvalue.py:
# Remove rows where any element is missing
data_cleaned_row = data_cleaned.dropna()
# Save the cleaned data back to a CSV file
# data_cleaned_row.to_csv('Cleaned_Final_Data.csv', index=False) # Un-comment this to actually save CSV file
# Check if there are still any missing values and display basic info of the cleaned data
missing_check = data_cleaned_row.isnull().sum()
info_cleaned = data_cleaned_row.info()
print("Data cleaned successfully. Here's the info of the cleaned dataset:")
print(info_cleaned)
print("\nCheck for any remaining missing values:")
print(missing_check)


## DATA PREPROCESSING
# normalisation.py:
# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Select columns to scale, assuming you want to scale all numerical columns
numerical_cols = data_cleaned_row.select_dtypes(include=['float64', 'int64']).columns
# Fit and transform the data
data_cleaned_row[numerical_cols] = scaler.fit_transform(data_cleaned_row[numerical_cols])
normalised_data = data_cleaned_row # Rename reference
# Save the normalized data to a new CSV file
# normalised_data.to_csv('Normalized_Cleaned_Final_Data.csv', index=False) # Un-comment this to actually save CSV file
# Display the first few rows of the normalized data
print(normalised_data.head())

# final_preprocessing.py:
# Step 1: Convert 'Date' column to datetime format
normalised_data['Date'] = pd.to_datetime(normalised_data['Date'])
# Step 2: Encode categorical variables using LabelEncoder
categorical_columns = ['Duplexing Type', 'Site Id', 'Sector', 'Sector id']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    normalised_data[column] = le.fit_transform(normalised_data[column])
    label_encoders[column] = le  # Store label encoder for each column to use later for inverse transform
# Step 3: Check for missing values (already claimed no missing values, but double-checking)
if normalised_data.isnull().sum().sum() > 0:
    normalised_data = normalised_data.fillna(normalised_data.mean())  # Filling missing values with mean if any
# Step 4: Check normalization and standardize if necessary
# Since data is already normalized, we'll verify this by checking if all values are between 0 and 1
if (normalised_data.select_dtypes(include=['float64', 'int64']).min().min() < 0) or (normalised_data.select_dtypes(include=['float64', 'int64']).max().max() > 1):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numeric_columns = normalised_data.select_dtypes(include=['float64', 'int64']).columns
    normalised_data[numeric_columns] = scaler.fit_transform(normalised_data[numeric_columns])
    preprocessing_data = normalised_data # Rename reference
# Step 5: Dimensionality reduction (optional, uncomment below lines to apply PCA)
# pca = PCA(n_components=10)  # Adjust components based on variance or desired feature reduction
# principal_components = pca.fit_transform(preprocessing_data.select_dtypes(include=['float64', 'int64']))
# preprocessing_data = pd.DataFrame(data=principal_components)
# Display updated data
print(preprocessing_data.head())
print(preprocessing_data.describe())
# Save the pre-processed data
# preprocessing_data.to_csv('Preprocessed_Data.csv', index=False) # Un-comment this to actually save CSV file


## FEATURES ENGINEER
# removingduplexing_type.py: 
# Dropping the 'Duplexing Type' column
remove_duplex_data = preprocessing_data.drop(['Duplexing Type', 'Number of CSFB to WCDMA'], axis=1)
# Save the updated dataframe to a new CSV file
# new_file_path = 'Updated_Preprocessed_Data.csv'                   # Un-comment this to actually save CSV file
# remove_duplex_data.to_csv(new_file_path, index=False)             # Un-comment this to actually save CSV file
# print("Updated dataset saved as 'Updated_Preprocessed_Data.csv'") # Un-comment this to actually save CSV file

# removingweakcorrelations.py:
# List of columns to remove
columns_to_remove = [
    'PS E-UTRAN RRC Setup successful Ratio (%)',
    'PS E-UTRAN RAB Setup Success Rate (%)',
    'DC_E_ERBS_EUTRANCELLFDD.pmCellDowntimeMan',
    'DC_E_ERBS_EUTRANCELLFDD.pmPagDiscarded',
    'Intra Frequency Handover Success Rate (%)',
    'DC_E_ERBS_EUTRANCELLFDD.pmCellDowntimeAuto',
    'Average User Throughput in DownLink (Mbps)'
]
# Drop these columns from the dataframe
remove_weak_correlations_data = remove_duplex_data.drop(columns=columns_to_remove, errors='ignore')
# Save the cleaned dataset to a new CSV file
# remove_weak_correlations_data.to_csv('Removed_weakcorrelations.csv', index=False) # Un-comment this to actually save CSV file
# Print the remaining columns after removal
print("Remaining columns after removal:", remove_weak_correlations_data.columns)

# correlationsALS1.py:
# Drop non-numeric columns for correlation analysis
numeric_data = remove_weak_correlations_data.select_dtypes(include=['float64', 'int64'])
# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()
# Set up the matplotlib figure
plt.figure(figsize=(13, 8))
# Create a heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
# Add title to the heatmap
plt.title('Feature Correlation Matrix for 5G Network Entities', fontsize=16)
# Display the plot
plt.show()

# feature_engineering.py:
# 1. Combined DL and UL Traffic: DL Traffic + UL Traffic
remove_weak_correlations_data['Combined_DL_UL_Traffic'] = remove_weak_correlations_data['DL Traffic (GB)'] + remove_weak_correlations_data['UL Traffic (GB)']
# 2. Total Throughput per User: Total User Throughput divided by the number of users 
remove_weak_correlations_data['Total_Throughput_per_User'] = (remove_weak_correlations_data['DL User Throughput(Mbps)_NUM'] + 
                                           remove_weak_correlations_data['Average User Throughput in UpLink (Mbps)_NUM']) / remove_weak_correlations_data['Maximum Number of Users in a Cell']
# 3. Traffic per PDCP Throughput: Total Traffic divided by total PDCP throughput
remove_weak_correlations_data['Traffic_per_PDCP_Throughput'] = remove_weak_correlations_data['Total Traffic (GB)'] / (
    remove_weak_correlations_data['Average DL PDCP Throughput (Mbit/s)'] + remove_weak_correlations_data['Average UL PDCP Throughput (Mbit/s)'])
# List of the new features to be normalized
new_features = ['Combined_DL_UL_Traffic', 'Total_Throughput_per_User', 'Traffic_per_PDCP_Throughput']
# Apply MinMaxScaler to normalize the new features
scaler = MinMaxScaler()
normalised_new_features_data = remove_weak_correlations_data # Rename reference
normalised_new_features_data[new_features] = scaler.fit_transform(remove_weak_correlations_data[new_features])
# Save the updated and normalized dataset to a new CSV file
# normalised_new_features_data.to_csv('normalized_dataset_with_new_features.csv', index=False) # Un-comment this to actually save CSV file
# Display the first few rows of the normalized dataset
print(normalised_new_features_data.head())

# correlationsTargetclass.py:
# Remove non-numeric columns (such as date)
numeric_df = normalised_new_features_data.select_dtypes(include=[float, int])
# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()
# Plotting the heatmap for correlations with 'Total Traffic (GB)'
plt.figure(figsize=(13, 8))
sns.heatmap(correlation_matrix[['Total Traffic (GB)']].sort_values(by='Total Traffic (GB)', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap with Total Traffic (GB)')
plt.show()