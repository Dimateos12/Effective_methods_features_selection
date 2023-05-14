from sklearn.model_selection import StratifiedKFold
from config.load_config import load_config

config = load_config("my_configuration.yaml")


def cross_validation(n_splits=config["k_fold"], random_state=None):
    """
    Splits the data into stratified folds using `StratifiedKFold`.

    Parameters:
    n_splits (int): Number of folds to be created.
    random_state (int or None): Seed value for random number generation.

    Returns:
    tuple: A tuple containing two lists of numpy arrays.
           The first list contains the training indices for each fold,
           and the second list contains the corresponding test indices.

    """
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    return skf
