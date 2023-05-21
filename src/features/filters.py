# import mdfs
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
# from skrebate import ReliefF
from features.UtestFeatureSelection import utest
from config.load_config import load_config
from ReliefF import ReliefF

config = load_config("my_configuration.yaml")


##DODAC ZWRACANIE INDEKSOW DO KAZDEGO FILTRA
def filters(x, y, module, num_of_features=config['n_features']):
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


    Returns:
    np.ndarray: The selected features as a numpy array of shape
        (n_samples, num_of_features).
    """

    if module == "ReliefF":
        print("Trwa selekcja cech metoda ReliefF....")
        rf = ReliefF(n_features_to_keep=num_of_features, n_neighbors=5)
        features = rf.fit_transform(x, y)
        selected_feature_indices = rf.top_features[:rf.n_features_to_keep]
        return selected_feature_indices
    elif module == "Mrmr":
        X = pd.DataFrame(x)
        y = pd.Series(y)
        selected_features_names = mrmr_classif(X=X, y=y, K=num_of_features)
        print(selected_features_names)
        features = X[selected_features_names]
        return features.index
    elif module == "U-test":
        X_feature = utest(x, y, )
        return X_feature
    # elif module == "MDFS":
    #     y = y.astype(np.int32)
    #     mdfs_feature = mdfs.run(x, y)
    #     return mdfs_feature.copy()

    # https://github.com/biocsuwb/EnsembleFS-package/blob/main/R/fs.mdfs.1D.R
