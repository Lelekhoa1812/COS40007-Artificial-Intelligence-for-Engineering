import pandas as pd

def remove_highly_correlated_features(df, threshold=0.95):
    """
    Remove one of the features from any pair of features that have a correlation above the given threshold.
    
    Parameters:
    df (DataFrame): The input dataframe
    threshold (float): The correlation threshold above which features are considered too similar
    
    Returns:
    DataFrame: The reduced dataframe with highly correlated features removed
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[float, int])
    
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr().abs()

    # Create a set to hold the names of columns to drop
    to_drop = set()

    # Iterate through the correlation matrix and find highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > threshold and correlation_matrix.columns[j] not in to_drop:
                to_drop.add(correlation_matrix.columns[i])

    # Drop the highly correlated columns
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced

# Example usage with a CSV file:
file_path = r"C:\Users\Saw\Desktop\data\feature_engineered(DL traffic)\Preprocessed_Data.csv"
data = pd.read_csv(file_path)

# Remove highly correlated features with a threshold of 0.95
reduced_data = remove_highly_correlated_features(data, threshold=0.95)

# Save the reduced dataframe to a new file if needed
reduced_data.to_csv(r'C:\Users\Saw\Desktop\data\feature_engineered(DL traffic)\reduced_data.csv', index=False)

# Show the reduced data
print(reduced_data.head())
