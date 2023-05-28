import numpy as np
from scipy.stats import mannwhitneyu

def utest(X, y, num_features ,alpha=0.05):
    """
    Wybiera cechy z użyciem testu U-Manna.

    Parametry:
    X (array-like): Macierz cech, gdzie wiersze to próbki, a kolumny to cechy.
    y (array-like): Wektor wartości docelowych.
    alpha (float): Poziom istotności testu (domyślnie 0.05).
    num_features (int): Liczba najważniejszych cech do wybrania (domyślnie 100).

    Zwraca:
    selected_features (list): Lista indeksów wybranych cech.

    """

    num_all_features = X.shape[1]
    p_values = []

    for i in range(num_all_features):
        feature_values = X[:, i]
        stat, p_value = mannwhitneyu(feature_values[y == 0], feature_values[y == 1], alternative='two-sided')
        p_values.append((i, p_value))

    # Sortowanie p-wartości w porządku rosnącym
    p_values.sort(key=lambda x: x[1])

    selected_features = [idx for idx, _ in p_values[:num_features]]

    return selected_features