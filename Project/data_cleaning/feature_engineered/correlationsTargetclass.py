import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path =  r"C:\Users\Saw\Desktop\data\feature_engineered\normalized_dataset_with_new_features.csv"  # Replace with the path to your dataset
df = pd.read_csv(file_path)


# Remove non-numeric columns (such as date)
numeric_df = df.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the heatmap for correlations with 'Total Traffic (GB)'
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[['Total Traffic (GB)']].sort_values(by='Total Traffic (GB)', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)

plt.title('Correlation Heatmap with Total Traffic (GB)')
plt.show()