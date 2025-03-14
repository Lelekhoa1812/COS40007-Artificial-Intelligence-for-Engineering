import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simps

def compute_statistical_features(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Define the columns to compute features on
    feature_columns = df.columns[2:20]  # Adjust the indices as needed for your columns
    
    # Prepare a list to hold all features DataFrames
    features_list = []

    # Compute features per minute (60 frames per minute assumed)
    frame_groups = df.groupby(df.index // 60)  # Grouping every 60 rows together assuming these are consecutive

    for name, group in frame_groups:
        minute_features = {
            'Frame Start': group['Frame'].iloc[0],
            'Frame End': group['Frame'].iloc[-1],
            'Class': group['Class'].iloc[0]  # Preserving the Class of the first frame in the minute
        }
        
        for column in feature_columns:
            values = group[column]
            minute_features[f'{column} Mean'] = values.mean()
            minute_features[f'{column} Std'] = values.std()
            minute_features[f'{column} Min'] = values.min()
            minute_features[f'{column} Max'] = values.max()
            minute_features[f'{column} AUC'] = simps(values)
            peaks, _ = find_peaks(values)
            minute_features[f'{column} Peaks'] = len(peaks)
        
        features_list.append(pd.DataFrame([minute_features]))

    # Combine all the DataFrames in the list into a single DataFrame
    features_df = pd.concat(features_list, ignore_index=True)

    # Save the extended dataset with features
    features_df.to_csv('minute_features.csv', index=False)
    return features_df

# Path to your CSV file
file_path = '20columns.csv'
df_features = compute_statistical_features(file_path)
print(df_features.head())  # Show a preview of the results
