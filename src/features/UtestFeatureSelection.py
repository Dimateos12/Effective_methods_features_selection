from scipy.stats import mannwhitneyu
import pandas as pd


def utest(X, y, k=100):
    """
    Select top k features using the Mann-Whitney U-test.

    Parameters:
        X (array-like): The input data matrix.
        y (array-like): The target variable vector.
        k (int): The number of features to select. Default is 100.

    Returns:
        (list): The indices of the selected features.
    """
    n_features = X.shape[1]
    p_values = []

    for i in range(n_features):
        stat, p = mannwhitneyu(X[:, i], y, alternative='two-sided')
        p_values.append(p)

    df = pd.DataFrame({'index': range(n_features), 'p_value': p_values})
    df = df.sort_values(by=['p_value'], ascending=True).iloc[:k]

    return df['index'].tolist()
