"""Test the TFT model"""

import unittest

import pandas as pd
import tensorflow as tf

from tfts import AutoConfig, AutoModel, KerasTrainer
from tfts.data import TimeSeriesSequence, get_data
from tfts.models.tft import TFTransformer, TFTransformerConfig


class TFTransformerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        custom_model_config = TFTransformerConfig(
            encoder_input_dim=5,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
            ffn_intermediate_size=256,
            max_position_embeddings=512,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
        )
        model = TFTransformer(predict_sequence_length, config=custom_model_config)

        x = tf.random.normal([2, 16, 5])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

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

        config = AutoConfig.for_model("tft")
        config.encoder_input_dim = ts_sequence[0][0].shape[-1]

        model = AutoModel.from_config(config, predict_sequence_length=predict_sequence_length)
        trainer = KerasTrainer(model)
        trainer.train(ts_sequence, epochs=1)


if __name__ == "__main__":
    unittest.main()
