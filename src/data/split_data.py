def split_data(df, target_col):
    """
    Splits the input dataframe into X (predictors) and y (target variable)
    dataframes based on the specified target column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to be split.
    target_col : str
        The name of the column containing the target variable.

    Returns:
    --------
    tuple of pandas.DataFrame
        A tuple containing two dataframes: the first is X, the second is y.
    """

    # Make a copy of the original dataframe to avoid modifying it
    df_copy = df.copy()

    # Extract the target variable column
    y = df_copy.pop(target_col)

    # Return the predictors (X) and target variable (y)
    return df_copy, y
