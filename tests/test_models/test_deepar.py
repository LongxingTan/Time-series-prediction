import unittest

import pandas as pd
import tensorflow as tf

from tfts import AutoConfig, AutoModel, KerasTrainer, Trainer
from tfts.data import TimeSeriesSequence, get_data
from tfts.models.deepar import DeepAR


class DeepARTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = DeepAR(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, predict_sequence_length, 3])
        loc, scale = model(x)
        self.assertEqual(loc.shape, (2, predict_sequence_length, 1))
        self.assertEqual(scale.shape, (2, predict_sequence_length, 1))

    def test_train(self):
        data = get_data(name="ar", seasonality=10.0, timesteps=40, n_series=10, seed=42)
        data["static"] = 2
        data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")

        train_sequence_length = 16
        predict_sequence_length = 4

        ts_sequence = TimeSeriesSequence(
            data=data,
            time_idx="time_idx",
            target_column="value",
            train_sequence_length=train_sequence_length,
            predict_sequence_length=predict_sequence_length,
            batch_size=16,
            group_column=["series"],  # Group by series ID
        )

        config = AutoConfig.for_model("deep_ar")

        model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
        trainer = KerasTrainer(model)
        trainer.train(ts_sequence, epochs=1)
