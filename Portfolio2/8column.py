import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('all_data.csv')

# Select the specified columns
columns_to_keep = ['Frame', 'Class', 'Neck x', 'Neck y', 'Neck z', 'Head x', 'Head y', 'Head z']
filtered_df = df[columns_to_keep]

# Save the filtered data to a new CSV file
filtered_df.to_csv('8columns.csv', index=False)
