from itertools import combinations
import numpy as np

def calculate_similarity(feature_set1, feature_set2):
    feature_set1 = np.atleast_1d(feature_set1).tolist()
    feature_set2 = np.atleast_1d(feature_set2).tolist()

    intersection_size = len([x for x in feature_set1 if x in feature_set2])
    min_size = min(len(feature_set1), len(feature_set2))

    similarity = (
        intersection_size
        - len(feature_set1) * len(feature_set2) / min_size
        - max(0, len(feature_set1) + len(feature_set2) - min_size)
    )

    return similarity

def calculate_asm(feature_sets):
    c = len(feature_sets)  # Liczba zestawów cech
    similarity_sum = 0

    # Obliczanie podobieństwa dla wszystkich par zestawów cech
    for i in range(c - 1):
        for j in range(i + 1, c):
            similarity = calculate_similarity(feature_sets[i], feature_sets[j])
            similarity_sum += similarity

    asm = 2 * similarity_sum / (c * (c - 1))
    return asm
