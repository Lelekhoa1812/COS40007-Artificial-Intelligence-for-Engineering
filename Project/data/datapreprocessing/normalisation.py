import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
file_path = 'Cleaned_Final_Data.csv'
data = pd.read_csv(file_path)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Select columns to scale, assuming you want to scale all numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Fit and transform the data
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Save the normalized data to a new CSV file
data.to_csv('Normalized_Cleaned_Final_Data.csv', index=False)

# Display the first few rows of the normalized data
print(data.head())
