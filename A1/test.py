import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def load_and_train_model(filename, feature_cols, target_col):
    try:
        # Load dataset
        df = pd.read_csv(filename)
        # Ensure all required columns are present
        missing_columns = [col for col in feature_cols + [target_col] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

        
        # Separate features and target variable
        X = df[feature_cols]  # Features
        y = df[target_col]  # Target variable
        
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

        # Create Decision Tree classifier object
        clf = DecisionTreeClassifier()

        # Train Decision Tree Classifier
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return accuracy
    except KeyError as e:
        print(f"KeyError: {e}. Check if the columns are correctly named in {filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define columns for each dataset
feature_cols_converted = [
    'Hardness', 'Sulfate', 'Conductivity', 'Organic_carbon', 'pH_Class'
]
target_col = 'Potability'  # Ensure this is the correct column name

# List of files
files = {
    "converted_water_potability.csv": feature_cols_converted,
    "normalised_water_potability.csv": feature_cols_converted,
    "features_water_potability.csv": feature_cols_converted,
    "selected_features_water_potability.csv": feature_cols_converted,
    "selected_converted_water_potability.csv": feature_cols_converted
}

# Evaluate models
for file, features in files.items():
    print(f"Evaluating model for {file}:")
    accuracy = load_and_train_model(file, features, target_col)
    if accuracy is not None:
        print(f"Accuracy for {file}: {accuracy:.4f}")
