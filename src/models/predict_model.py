import numpy as np


def predict_model(model, X_test):
    """
    Function that performs predictions for a given machine learning model and a test dataset.

    Parameters:
    model : sklearn model object
        A trained machine learning model object from the scikit-learn library.
    X_test : numpy array
        A numpy array containing the test data.

    Returns:
    y_pred : numpy array
        A numpy array containing the predicted values for the test data.
    """

    # Perform predictions on the test data
    y_pred = model.predict(X_test)

    # Return the predictions
    return y_pred
