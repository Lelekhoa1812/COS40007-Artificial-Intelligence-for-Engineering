import pandas as pd

# Load the dataset
file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(DL traffic)\reduced_data.csv'
data = pd.read_csv(file_path)

# Dropping the specified columns
data_dropped = data.drop([
    'DL Traffic (GB)',
    'Date',
    'Duplexing Type',
    'Number of CSFB to WCDMA',

], axis=1)

# Save the updated dataframe to a new CSV file with a different name
new_file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(DL traffic)\final_target_class.csv'
data_dropped.to_csv(new_file_path, index=False)

print("Updated dataset saved as 'final_target.csv'")
