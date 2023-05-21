import numpy as np
from scipy.stats import mannwhitneyu

def utest(X, y, alpha=0.05):
    """
    Wybiera cechy z użyciem testu U-Manna.

    Parametry:
    X (array-like): Macierz cech, gdzie wiersze to próbki, a kolumny to cechy.
    y (array-like): Wektor wartości docelowych.
    alpha (float): Poziom istotności testu (domyślnie 0.05).

    Zwraca:
    selected_features (list): Lista indeksów wybranych cech.

    """

    num_features = X.shape[1]
    selected_features = []

    for i in range(num_features):
        feature_values = X[:, i]
        stat, p_value = mannwhitneyu(feature_values[y == 0], feature_values[y == 1], alternative='two-sided')

        if p_value <= alpha:
            selected_features.append(i)

    return [idx for idx, _ in enumerate(selected_features)]
