import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
data_path = 'Normalized_Cleaned_Final_Data.csv'
data = pd.read_csv(data_path)

# Step 1: Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Step 2: Encode categorical variables using LabelEncoder
categorical_columns = ['Duplexing Type', 'Site Id', 'Sector', 'Sector id']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Store label encoder for each column to use later for inverse transform

# Step 3: Check for missing values (already claimed no missing values, but double-checking)
if data.isnull().sum().sum() > 0:
    data = data.fillna(data.mean())  # Filling missing values with mean if any

# Step 4: Check normalization and standardize if necessary
# Since data is already normalized, we'll verify this by checking if all values are between 0 and 1
if (data.select_dtypes(include=['float64', 'int64']).min().min() < 0) or (data.select_dtypes(include=['float64', 'int64']).max().max() > 1):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Step 5: Dimensionality reduction (optional, uncomment below lines to apply PCA)
# pca = PCA(n_components=10)  # Adjust components based on variance or desired feature reduction
# principal_components = pca.fit_transform(data.select_dtypes(include=['float64', 'int64']))
# data = pd.DataFrame(data=principal_components)

# Display updated data
print(data.head())
print(data.describe())

# Save the pre-processed data
data.to_csv('Preprocessed_Data.csv', index=False)
