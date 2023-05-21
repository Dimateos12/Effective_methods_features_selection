def calculate_asm(feature_sets):
    """
    Calculates the ASM for a list of feature sets.

    Args:
        feature_sets (list): List of feature sets, where each
        feature set is a set of feature indices.

    Returns:
        float: The ASM value.
    """
    feature_sets = [set([fs]) for fs in feature_sets]  # Convert integers to singleton sets
    c = len(feature_sets)
    m = len(feature_sets[0])  # Assumes all feature sets have the same length

    similarity_sum = 0
    for i in range(c - 1):
        for j in range(i + 1, c):
            intersection_size = len(feature_sets[i].intersection(feature_sets[j]))
            min_size = min(len(feature_sets[i]), len(feature_sets[j]))
            similarity = (
                    intersection_size
                    - len(feature_sets[i]) * len(feature_sets[j]) / min_size
                    - max(0, len(feature_sets[i]) + len(feature_sets[j]) - m)
            )
            similarity_sum += similarity

    asm = 2 * similarity_sum / (c * (c - 1))
    return asm
