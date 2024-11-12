import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and returns it as a pandas DataFrame.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(filepath)

def split_input_output(data: pd.DataFrame, target_col: str) -> tuple:
    """
    Splits the DataFrame into input features and output target.

    Parameters:
    data (pd.DataFrame): The complete DataFrame.
    target_col (str): The name of the target column.

    Returns:
    tuple: A tuple containing (X, y) where X is the input features and y is the output target.
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]
    print(f"Original data shape: {data.shape}")
    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}")
    return X, y

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int = None) -> tuple:
    """
    Splits the data into training and test sets.

    Parameters:
    X (pd.DataFrame): The input features.
    y (pd.Series): The output target.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split.

    Returns:
    tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"X train shape: {X_train.shape}")
    print(f"X test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def serialize_data(data: object, path: str) -> None:
    """
    Serializes the given data and saves it to the specified path.

    Parameters:
    data (object): The data to serialize.
    path (str): The path to save the serialized data.
    """
    joblib.dump(data, path)

def deserialize_data(filename: str) -> object:
    """
    Deserializes data from the specified filename in the interim directory and returns the data.

    Parameters:
    filename (str): The name of the file where the serialized data is stored.

    Returns:
    object: The deserialized data.
    """
    path = f"{filename}"
    return joblib.load(path)
