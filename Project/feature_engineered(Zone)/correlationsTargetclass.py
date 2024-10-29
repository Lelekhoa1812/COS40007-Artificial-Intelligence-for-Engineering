import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\Saw\Desktop\data\feature_engineered(Zone)\normalized_final_target_class_with_site_composite_features.csv'  # Replace with the path to your dataset
df = pd.read_csv(file_path)

# Remove non-numeric columns (such as date)
numeric_df = df.select_dtypes(include=[float, int])

# Specify the columns to drop
columns_to_drop = [
    
]

# Drop the specified columns
numeric_df = numeric_df.drop(columns=columns_to_drop, errors='ignore')

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the heatmap for correlations with 'Traffic_per_PDCP_Throughput'
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix['Site Id'].sort_values(ascending=False).to_frame(), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)

plt.title('Correlation Heatmap with Site Id')
plt.show()