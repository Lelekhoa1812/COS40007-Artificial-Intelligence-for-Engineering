import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with the actual path to your CSV file)
#file_path = r"C:\Users\Saw\Desktop\assignment2\dataanalysis\final_data_processing.csv"
file_path = r"C:\Users\Saw\Desktop\data\feature_engineered\Removed_weakcorrelations.csv"

data = pd.read_csv(file_path)

# Drop non-numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# Add title to the heatmap
plt.title('Feature Correlation Matrix for Housing Dataset', fontsize=16)

# Display the plot
plt.show()
