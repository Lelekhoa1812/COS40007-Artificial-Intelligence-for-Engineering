import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset (replace with the actual file path)
file_path = r"C:\Users\Saw\Desktop\data\feature_engineered\Removed_weakcorrelations.csv"
df_cleaned = pd.read_csv(file_path)

# 1. Combined DL and UL Traffic: DL Traffic + UL Traffic
df_cleaned['Combined_DL_UL_Traffic'] = df_cleaned['DL Traffic (GB)'] + df_cleaned['UL Traffic (GB)']

# 2. Total Throughput per User: Total User Throughput divided by the number of users 
df_cleaned['Total_Throughput_per_User'] = (df_cleaned['DL User Throughput(Mbps)_NUM'] + 
                                           df_cleaned['Average User Throughput in UpLink (Mbps)_NUM']) / df_cleaned['Maximum Number of Users in a Cell']

# 3. Traffic per PDCP Throughput: Total Traffic divided by total PDCP throughput
df_cleaned['Traffic_per_PDCP_Throughput'] = df_cleaned['Total Traffic (GB)'] / (
    df_cleaned['Average DL PDCP Throughput (Mbit/s)'] + df_cleaned['Average UL PDCP Throughput (Mbit/s)'])

# List of the new features to be normalized
new_features = ['Combined_DL_UL_Traffic', 'Total_Throughput_per_User', 'Traffic_per_PDCP_Throughput']

# Apply MinMaxScaler to normalize the new features
scaler = MinMaxScaler()
df_cleaned[new_features] = scaler.fit_transform(df_cleaned[new_features])

# Save the updated and normalized dataset to a new CSV file
df_cleaned.to_csv(r'C:\Users\Saw\Desktop\data\feature_engineered\normalized_dataset_with_new_features.csv', index=False)

# Display the first few rows of the normalized dataset
print(df_cleaned.head())
