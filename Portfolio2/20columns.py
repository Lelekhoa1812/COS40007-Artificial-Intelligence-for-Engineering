import pandas as pd
import numpy as np

def load_and_process_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Calculate the root mean square (RMS) for different column sets
    for prefix in ['Neck', 'Head']:
        x_col, y_col, z_col = f'{prefix} x', f'{prefix} y', f'{prefix} z'
        df[f'RMS x and y {prefix}'] = np.sqrt((df[x_col]**2 + df[y_col]**2) / 2)
        df[f'RMS y and z {prefix}'] = np.sqrt((df[y_col]**2 + df[z_col]**2) / 2)
        df[f'RMS z and x {prefix}'] = np.sqrt((df[z_col]**2 + df[x_col]**2) / 2)
        df[f'RMS x, y, z {prefix}'] = np.sqrt((df[x_col]**2 + df[y_col]**2 + df[z_col]**2) / 3)
    
        # Calculate Roll and Pitch
        df[f'Roll {prefix}'] = 180 * np.arctan2(df[y_col], np.sqrt(df[x_col]**2 + df[z_col]**2)) / np.pi
        df[f'Pitch {prefix}'] = 180 * np.arctan2(df[x_col], np.sqrt(df[y_col]**2 + df[z_col]**2)) / np.pi

    # Save the extended dataset to a new CSV file
    df.to_csv('20columns.csv', index=False)
    print("Data processing complete and saved to 20columns.csv.")

# Path to your CSV file
file_path = '8columns.csv'
load_and_process_data(file_path)
