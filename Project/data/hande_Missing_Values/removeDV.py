import pandas as pd

# Load the dataset from a CSV file
file_path = 'LTE_KPI.csv'
data = pd.read_csv(file_path)

# Identify columns containing "#DIV/0"
div0_columns = [col for col in data.columns if data[col].astype(str).str.contains("#DIV/0").any()]

# Remove the identified columns
cleaned_data = data.drop(columns=div0_columns)

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleanedDV.csv'
cleaned_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
