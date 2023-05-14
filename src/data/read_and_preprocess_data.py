import pandas as pd
import os
from config.load_config import load_config

config = load_config("my_configuration.yaml")


def read_and_preprocess_data(
        file_path=os.path.join(config["data_directory"], config["data_name"]),
        target_column=config["target_name"]):
    """
    Reads a CSV file from a given path, preprocesses it,
    and returns the preprocessed data.

    Parameters:
    ----------

    file_path: str: The path of the CSV file to be read.
    target_column: str: The name of the target column.
    num_columns: int: The number of columns to keep in the
        preprocessed data. Defaults to 1000.

    Returns:
    ----------

    tuple: A tuple containing the preprocessed data in
        the following format: (X, y, df).
    X (numpy.ndarray): The preprocessed feature matrix.
    y (numpy.ndarray): The preprocessed target vector.
    df (pandas.DataFrame): The original dataframe
        that was read from the CSV file.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()

    return X, y, df
