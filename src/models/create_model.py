from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators=None, max_depth=None):
    """
    Function that creates a random forest model with specified hyperparameters.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        Number of trees in the random forest.

    max_depth : int or None, optional (default=None)
        Maximum depth of a tree.

    Returns
    -------
    model : RandomForestClassifier
        Random forest model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return model