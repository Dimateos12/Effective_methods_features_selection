import mdfs
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from skrebate import ReliefF
from features.UtestFeatureSelection  import UTestFeatureSelection

def filters(x, y, num_of_features, module, df):
    """
    Selects a given number of features from the input data using the
    specified feature selection algorithm.

    Parameters:
    x (np.ndarray): Array of shape (n_samples, n_features)
        containing the input features.
    y (np.ndarray): Array of shape (n_samples,) containing the target labels.
    num_of_features (int): The number of features to select.
    module (str): The name of the feature selection algorithm to use.
        Possible values: "ReliefF", "Mrmr", "U-test", "MDFS".
    df (pd.DataFrame): The input data as a pandas DataFrame object.

    Returns:
    np.ndarray: The selected features as a numpy array of shape
        (n_samples, num_of_features).
    """

    if module == "ReliefF":
        rf = ReliefF(n_features_to_select=num_of_features)
        return rf.fit_transform(x, y)
    elif module == "Mrmr":
        X = pd.DataFrame(x)
        y = pd.Series(y)
        selected_features_names = mrmr_classif(X=X, y=y, K=num_of_features)
        return X[selected_features_names]
    elif module == "U-test":
        utest = UTestFeatureSelection(df, "class", num_of_features)
        return utest.transform()
    elif module == "MDFS":
        y = y.astype(np.int32)
        mdfs_feature = mdfs.run(x, y)
        return mdfs_feature.copy()

    # https://github.com/biocsuwb/EnsembleFS-package/blob/main/R/fs.mdfs.1D.R
