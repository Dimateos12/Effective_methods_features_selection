from sklearn.model_selection import StratifiedKFold


def cross_validation(X, y, n_splits=5, random_state=None):
    """
    Splits the data into stratified folds using `StratifiedKFold`.

    Parameters:
    X (numpy.ndarray): Input feature matrix.
    y (numpy.ndarray): Target variable array.
    n_splits (int): Number of folds to be created.
    random_state (int or None): Seed value for random number generation.

    Returns:
    tuple: A tuple containing two lists of numpy arrays.
           The first list contains the training indices for each fold,
           and the second list contains the corresponding test indices.

    """
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    train_indices_list = []
    test_indices_list = []

    for train_indices, test_indices in skf.split(X, y):
        train_indices_list.append(train_indices)
        test_indices_list.append(test_indices)

    return train_indices_list, test_indices_list
