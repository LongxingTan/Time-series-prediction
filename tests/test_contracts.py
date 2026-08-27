import unittest

import tensorflow as tf

from tfts.contracts import ClassificationTaskConfig, ForecastTaskConfig, TaskType, TimeSeriesBatch


class TimeSeriesBatchTest(unittest.TestCase):
    def test_mapping_only_accepts_canonical_names(self):
        batch = TimeSeriesBatch.from_inputs({"past_values": tf.zeros([2, 8, 1])})
        self.assertEqual(batch.past_values.shape, (2, 8, 1))

        with self.assertRaisesRegex(ValueError, "Unknown TimeSeriesBatch fields"):
            TimeSeriesBatch.from_inputs({"x": tf.zeros([2, 8, 1])})

    def test_positional_inputs_are_rejected(self):
        values = tf.zeros([2, 8, 1])
        past_features = tf.zeros([2, 8, 2])
        future_features = tf.zeros([2, 4, 3])

        with self.assertRaisesRegex(ValueError, "Positional time-series inputs are ambiguous"):
            TimeSeriesBatch.from_inputs((values, past_features, future_features))

    def test_imputation_requires_observed_mask(self):
        batch = TimeSeriesBatch(tf.zeros([2, 8, 1]))
        with self.assertRaisesRegex(ValueError, "requires past_observed_mask"):
            batch.validate_for("imputation")

    def test_temporal_feature_lengths_are_validated(self):
        batch = TimeSeriesBatch(
            past_values=tf.zeros([2, 8, 1]),
            past_categorical_features=tf.zeros([2, 7, 2], tf.int32),
        )
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "time length mismatch"):
            batch.validate_for("forecasting")

    def test_future_feature_horizons_must_match(self):
        batch = TimeSeriesBatch(
            past_values=tf.zeros([2, 8, 1]),
            future_time_features=tf.zeros([2, 4, 2]),
            future_categorical_features=tf.zeros([2, 3, 1], tf.int32),
        )
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "horizon mismatch"):
            batch.validate_for("forecasting")


class TaskConfigTest(unittest.TestCase):
    def test_configs_are_validated_and_json_friendly(self):
        config = ForecastTaskConfig(task="forecasting", prediction_length=4, head="quantile", quantiles=[0.1, 0.5, 0.9])
        self.assertEqual(config.task, TaskType.FORECASTING)
        self.assertEqual(config.quantiles, (0.1, 0.5, 0.9))
        self.assertEqual(config.to_dict()["task"], "forecasting")

        classification = ClassificationTaskConfig(hidden_units=[32, 16])
        self.assertEqual(classification.hidden_units, (32, 16))

    def test_invalid_values_fail_at_the_configuration_boundary(self):
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            ForecastTaskConfig(head="quantile", quantiles=(0.5, 0.1))
        with self.assertRaisesRegex(ValueError, "dropout"):
            ClassificationTaskConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()
