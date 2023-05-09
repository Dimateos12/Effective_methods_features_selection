import unittest

import pandas as pd
from src.data.split_data import split_data

class TestSplitData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "Y": [10, 11, 12]}
        )

    def test_split_X_Y(self):
        X, Y = split_data(self.df, "Y")
        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(Y.shape, (3,))
        self.assertListEqual(list(X.columns), ["A", "B", "C"])
        self.assertEqual(Y.name, "Y")
        self.assertListEqual(list(X.index), [0, 1, 2])
        self.assertListEqual(list(Y.index), [0, 1, 2])
        self.assertTrue(all(X["A"] == [1, 2, 3]))
        self.assertTrue(all(Y == [10, 11, 12]))
