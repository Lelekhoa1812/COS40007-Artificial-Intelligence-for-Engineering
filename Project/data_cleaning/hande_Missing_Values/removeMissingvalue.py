import pandas as pd

# Load the dataset
data = pd.read_csv('cleanedDV.csv')

# Remove columns with all missing values
data_cleaned = data.dropna(axis=1, how='all')

# Convert the 'Date' column to datetime format
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])

# Save the cleaned data to a new CSV file
data_cleaned.to_csv('Removed_Missing_Values.csv', index=False)

print("Cleaned data has been saved. Here's the info of the cleaned dataset:")
print(data_cleaned.info())
