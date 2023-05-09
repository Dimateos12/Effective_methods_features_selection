from sklearn import preprocessing


def scaler(X):
    """
    Normalize values in columns of X using the MinMaxScaler method.

    Parameters
    ----------
       X : array-like of shape (n_samples, n_features)
            Input feature values to be normalized.


    Returns
    -------
       x_scaled : ndarray of shape (n_samples, n_features)
        Normalized feature values.

    """
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(X)
    return x_scaled
