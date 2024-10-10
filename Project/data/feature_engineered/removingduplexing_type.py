import pandas as pd

# Load the dataset
file_path = r'C:\Users\Saw\Desktop\data\feature_engineered\Preprocessed_Data.csv'
data = pd.read_csv(file_path)

# Dropping the 'Duplexing Type' column
data_dropped = data.drop(['Duplexing Type', 'Number of CSFB to WCDMA'], axis=1)
# Save the updated dataframe to a new CSV file
new_file_path = r"C:\Users\Saw\Desktop\data\feature_engineered\Updated_Preprocessed_Data.csv"
data_dropped.to_csv(new_file_path, index=False)

print("Updated dataset saved as 'Updated_Preprocessed_Data.csv'")
