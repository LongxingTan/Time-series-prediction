"""Feature boundary tests for the canonical transforms."""

import unittest

import numpy as np
import pandas as pd

from tfts.data.features import FeaturePipeline, LagTransform, RollingTransform
from tfts.data.schema import Schema


class HistoryFeaturesTest(unittest.TestCase):
    def test_grouped_unsorted_input_never_fills_from_another_series(self):
        frame = pd.DataFrame(
            {"time": [2, 0, 1, 2, 0, 1], "group": ["a"] * 3 + ["b"] * 3, "value": [3.0, 1.0, np.nan, 30.0, 10.0, 20.0]}
        )
        schema = Schema("time", ("value",), group_cols=("group",))
        result = (
            FeaturePipeline([LagTransform("value", [1]), RollingTransform("value", [2])])
            .fit_transform(frame, schema)
            .frame
        )
        np.testing.assert_allclose(result.value_lag_1, [np.nan, 1, np.nan, np.nan, 10, 20], equal_nan=True)
        self.assertTrue(result.value_roll_2_mean.iloc[:5].isna().all())
        self.assertEqual(result.value_roll_2_mean.iloc[5], 15)

    def test_invalid_history_configuration(self):
        for factory in (
            lambda: LagTransform("value", [-1]),
            lambda: RollingTransform("value", [0]),
            lambda: RollingTransform("value", [2], functions=["unknown"]),
            lambda: RollingTransform("value", [2], shift=-1),
        ):
            with self.assertRaises(ValueError):
                factory()
