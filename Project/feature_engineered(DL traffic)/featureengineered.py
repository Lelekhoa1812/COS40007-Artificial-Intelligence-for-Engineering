import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(DL traffic)\final_target_class.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# 1. Throughput Efficiency
df['Throughput Efficiency'] = (df['Average DL PDCP Throughput (Mbit/s)'] + df['Average UL PDCP Throughput (Mbit/s)']) / df['DL User Throughput(Mbps)_DENOM']

# 2. Handover Success Rate
df['Handover Success Rate'] = (df['Intra Frequency Handover Success Rate (%)'] + df['Inter Frequency Handover Success Rate (%)']) / 2

# 3. User Density Impact
df['User Density Impact'] = (df['Maximum Number of Users in a Cell'] * df['Scell Traffic Volume']) / df['DL User Throughput(Mbps)_DENOM']

# Select the features you want to normalize (including the new composite features)
features_to_normalize = ['Throughput Efficiency', 'Handover Success Rate', 'User Density Impact']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected features
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Save the normalized dataset
output_file_path = 'normalized_final_target_class_with_composite_features.csv'  # Change the path if necessary
df.to_csv(output_file_path, index=False)

print(f'Normalized dataset saved to {output_file_path}')
