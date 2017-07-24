import unittest
from lib.features import features, append_feature
import pandas as pd
import numpy as np
from functools import reduce

dates = pd.date_range('20130101', periods=20)

fixture = pd.DataFrame([], index=dates)
fixture['Adj. Close'] = [
    100, 150, 200, 100, 300, 250, 200, 100, 150, 120, 170, 190, 200, 210, 250,
    178, 111, 190, 192, 160
]

fixture['Adj. Volume'] = [x * 10 for x in range(0, 20)]


class TestStringMethods(unittest.TestCase):
    def test_use_adj_close(self):
        actual = features(fixture)['Adj. Close']
        expected = fixture['Adj. Close']

        check_panda_equality(self, actual, expected)

    def test_use_adj_volume(self):
        actual = features(fixture)['Adj. Volume']
        expected = fixture['Adj. Volume']

        check_panda_equality(self, actual, expected)

    def test_use_last_14_days_avg_diff(self):
        actual = features(fixture)['Last 14 days diff'].values[-1]

        sum = reduce((lambda a, c: a + c), fixture['Adj. Close'][-15:-1])
        expected = (sum / 14) - fixture['Adj. Close'][-15]

        self.assertEqual(actual, expected)

    def test_append_feature(self):
        actual = append_feature(fixture, 50)

        dates = pd.date_range('20130101', periods=21)
        expected = pd.DataFrame([], index=dates)
        expected['Adj. Close'] = [
            100, 150, 200, 100, 300, 250, 200, 100, 150, 120, 170, 190, 200,
            210, 250, 178, 111, 190, 192, 160, 50
        ]
        expected['Adj. Volume'] = [x * 10 for x in range(0, 20)] + [190]

        check_panda_equality(self, actual, expected)


def check_panda_equality(self, actual, expected):
    isEqual = actual.equals(expected)
    if not isEqual:
        print('expected', expected)
        print('actual', actual)

    self.assertTrue(isEqual)
