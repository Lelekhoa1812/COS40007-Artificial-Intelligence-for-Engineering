import pandas as pd

# Load the dataset
data = pd.read_csv('Removed_Missing_Values.csv')

# Remove rows where any element is missing
data_cleaned = data.dropna()

# Save the cleaned data back to a CSV file
data_cleaned.to_csv('Cleaned_Final_Data.csv', index=False)

# Check if there are still any missing values and display basic info of the cleaned data
missing_check = data_cleaned.isnull().sum()
info_cleaned = data_cleaned.info()

print("Data cleaned successfully. Here's the info of the cleaned dataset:")
print(info_cleaned)
print("\nCheck for any remaining missing values:")
print(missing_check)
