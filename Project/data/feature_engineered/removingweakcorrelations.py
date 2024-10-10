import pandas as pd

# Load your dataset (update the file path as necessary)
file_path = r"C:\Users\Saw\Desktop\data\feature_engineered\Updated_Preprocessed_Data.csv"  # Replace with your dataset path
df = pd.read_csv(file_path)

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
df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv(r"C:\Users\Saw\Desktop\data\feature_engineered\Removed_weakcorrelations.csv", index=False)

# Print the remaining columns after removal
print("Remaining columns after removal:", df_cleaned.columns)
