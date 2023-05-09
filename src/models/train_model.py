from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
        X_train, y_train, n_estimators=100, max_depth=None, random_state=None):
    """
    Trains a random forest classifier on the input training data.

    Parameters:
    X_train : numpy array
        A numpy array containing the features for the training data.
    y_train : numpy array
        A numpy array containing the labels for the training data.
    n_estimators : int, optional
        The number of trees in the forest. Default is 100.
    max_depth : int or None, optional
        The maximum depth of the tree. Default is None.
    random_state : int or None, optional
        Seed for random number generation. Default is None.

    Returns:
    rf_model : RandomForestClassifier object
        A trained random forest classifier model.
    """

    # Initialize a random forest classifier with the specified
    # hyperparameters

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Train the model on the input data
    rf_model.fit(X_train, y_train)

    # Return the trained model
    return rf_model
