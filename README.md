import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    # Create a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Display the first few rows
    print("First few rows of the dataset:")
    print(df.head())
    
    # Check data types and missing values
    print("\nData Types and Missing Values:")
    print(df.info())
    print("\nAny Missing Values?")
    print(df.isnull().sum())

    # Clean the dataset (no missing values in Iris, but shown for process)
    df.fillna(method='ffill', inplace=True)  # Example of filling missing values
except Exception as e:
    print(f"An error occurred: {e}")
