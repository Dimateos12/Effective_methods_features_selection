import unittest
import pandas as pd
from cross_validation import cross_validation


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "B": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "C": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            }
        )

    def test_cross_validation(self, k=5):
        train_test_sets = cross_validation(self.df, k=k)

        # check number of sets
        self.assertEqual(len(train_test_sets), k)

        # check train/test set sizes
        num_rows = len(self.df.index)
        fold_size = num_rows // k
        for i in range(k):
            train_set, test_set = train_test_sets[i]
            expected_train_size = (k - 1) * fold_size
            expected_test_size = fold_size
            self.assertEqual(len(train_set.index), expected_train_size)
            self.assertEqual(len(test_set.index), expected_test_size)

            # check train/test set contents
            train_cols = set(self.df.columns)
            test_cols = set(self.df.columns)
            for j in range(k):
                if j == i:
                    test_cols -= set(train_set.columns)
                else:
                    train_cols -= set(test_set.columns)
            train_vals = set(train_set[train_cols].values.flatten())
            test_vals = set(test_set[test_cols].values.flatten())
            self.assertTrue(len(train_vals & test_vals) == 0)
