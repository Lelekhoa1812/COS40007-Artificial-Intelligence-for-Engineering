import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(Zone)\final_target_class.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Create the composite features for Site ID

# 1. Average Throughput per Site
average_throughput = df.groupby('Site Id')['DL User Throughput(Mbps)_DENOM'].mean().reset_index()
average_throughput.columns = ['Site Id', 'Average Throughput per Site']

# 2. Total Scell Traffic Volume per Site
total_traffic_volume = df.groupby('Site Id')['Scell Traffic Volume'].sum().reset_index()
total_traffic_volume.columns = ['Site Id', 'Total Scell Traffic Volume per Site']

# 3. Handover Success Rate per Site
handover_success_rate = df.groupby('Site Id').apply(lambda x: (x['Intra Frequency Handover Success Rate (%)'].mean() + 
                                                                x['Inter Frequency Handover Success Rate (%)'].mean()) / 2).reset_index()
handover_success_rate.columns = ['Site Id', 'Handover Success Rate per Site']

# Merge the new composite features back into the original dataframe
df = df.merge(average_throughput, on='Site Id', how='left')
df = df.merge(total_traffic_volume, on='Site Id', how='left')
df = df.merge(handover_success_rate, on='Site Id', how='left')

# Select the features to normalize (including the new composite features)
features_to_normalize = ['Average Throughput per Site', 'Total Scell Traffic Volume per Site', 'Handover Success Rate per Site']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected features
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Save the normalized dataset with new composite features
output_file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(Zone)\normalized_final_target_class_with_site_composite_features.csv'  # Change the path if necessary
df.to_csv(output_file_path, index=False)

print(f'Normalized dataset with Site ID composite features saved to {output_file_path}')
