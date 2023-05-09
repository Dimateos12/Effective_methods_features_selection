def train_random_forest(X_train, y_train, rf_model, n_estimators=100, max_depth=None, random_state=None):
    """
    Trains a random forest classifier on the input training data.

    Parameters:
    -----------
    X_train : numpy array
        A numpy array containing the features for the training data.
    y_train : numpy array
        A numpy array containing the labels for the training data.
    rf_model : RandomForestClassifier object
        A RandomForestClassifier object to be trained on the input data.
    n_estimators : int, optional (default=100)
        The number of trees in the forest.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree.
    random_state : int or None, optional (default=None)
        Seed for random number generation.

    Returns:
    --------
    rf_model : RandomForestClassifier object
        A trained random forest classifier model.
    """

    # Initialize a random forest classifier with the specified
    # hyperparameters
    rf_model.set_params(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Train the model on the input data
    rf_model.fit(X_train, y_train)

    # Return the trained model
    return rf_model
