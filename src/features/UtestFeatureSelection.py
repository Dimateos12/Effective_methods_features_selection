import pandas as pd
from scipy.stats import mannwhitneyu


class UTestFeatureSelection:
    """
    UTestFeatureSelection - Class for feature selection using Mann-Whitney U test.

    Attributes:
        data (pd.DataFrame): Input dataset for feature selection.
        target_col (str): Name of the target column in the dataset.
        number_of_features (int): Number of top features to select.
        significance_level (float): Significance level for the U test. Default is 0.05.

    Methods:
        fit(): Fit the feature selection model and compute the U
            statistics and p-values.
        transform(): Transform the dataset by selecting the top significant features.

    Example Usage:
        selector = UTestFeatureSelection(data, 'target', 10)
        selected_data = selector.transform()

    """

    def __init__(self, data, target_col, number_of_features, significance_level=0.05):
        """
        Initializes the UTestFeatureSelection object.

        Args:
            data (pd.DataFrame): Input dataset for feature selection.
            target_col (str): Name of the target column in the dataset.
            number_of_features (int): Number of top features to
                select.
            significance_level (float): Significance level for the U
                test. Default is 0.05.
        """
        self.data = data
        self.target_col = target_col
        self.significance_level = significance_level
        self.results = None
        self.significant_features = None
        self.number_of_features = number_of_features

    def fit(self):
        """
        Fit the UTestFeatureSelection model and compute the U statistics and p-values.
        """
        class_1 = self.data[self.data[self.target_col] == 1]
        class_2 = self.data[self.data[self.target_col] == 0]

        if class_1.shape[0] > class_2.shape[0]:
            class_1 = class_1.sample(class_2.shape[0], replace=True)
        elif class_2.shape[0] > class_1.shape[0]:
            class_2 = class_2.sample(class_1.shape[0], replace=True)

        u_stats = []
        p_values = []
        for col in self.data.columns[:-1]:
            if col == self.target_col:
                continue
            u, p = mannwhitneyu(class_1[col], class_2[col], alternative="two-sided")
            u_stats.append(u)
            p_values.append(p)

        self.results = pd.DataFrame(
            {
                "feature": self.data.columns[:-2],
                "U-statistic": u_stats,
                "p-value": p_values,
            }
        )
        self.results = self.results[self.results["feature"] != self.target_col]

        self.results = self.results.sort_values(by="p-value")

        self.significant_features = self.results[
            self.results["p-value"] < self.significance_level
        ]["feature"].tolist()

    def transform(self):
        """
        Transform the dataset by selecting the top significant features.

        Returns:
            pd.DataFrame: Dataset with the selected features.
        """
        self.fit()
        return self.data[self.significant_features[: self.number_of_features]]
