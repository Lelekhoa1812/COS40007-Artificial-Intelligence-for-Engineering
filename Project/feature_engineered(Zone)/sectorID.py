import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(Zone)\final_target_class.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Create the composite features for Sector ID

# 1. Average DL User Throughput per Sector
average_dl_throughput = df.groupby('Sector id')['DL User Throughput(Mbps)_DENOM'].mean().reset_index()
average_dl_throughput.columns = ['Sector id', 'Average DL User Throughput per Sector']

# 2. Total Traffic Volume per Sector
total_traffic_volume = df.groupby('Sector id')['Scell Traffic Volume'].sum().reset_index()
total_traffic_volume.columns = ['Sector id', 'Total Traffic Volume per Sector']

# 3. Average Handover Success Rate per Sector
handover_success_rate = df.groupby('Sector id').apply(lambda x: (x['Intra Frequency Handover Success Rate (%)'].mean() + 
                                                                x['Inter Frequency Handover Success Rate (%)'].mean()) / 2).reset_index()
handover_success_rate.columns = ['Sector id', 'Average Handover Success Rate per Sector']

# Merge the new composite features back into the original dataframe
df = df.merge(average_dl_throughput, on='Sector id', how='left')
df = df.merge(total_traffic_volume, on='Sector id', how='left')
df = df.merge(handover_success_rate, on='Sector id', how='left')

# Select the features to normalize (including the new composite features)
features_to_normalize = ['Average DL User Throughput per Sector', 'Total Traffic Volume per Sector', 'Average Handover Success Rate per Sector']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected features
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Save the normalized dataset with new composite features
output_file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(Zone)\normalized_final_target_class_with_sector_composite_features.csv'  # Change the path if necessary
df.to_csv(output_file_path, index=False)

print(f'Normalized dataset with Sector ID composite features saved to {output_file_path}')
