"""Test the TFT model"""

import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from tfts import AutoConfig, AutoModel, KerasTrainer, TimeSeriesBatch
from tfts.data import SequenceMaterializer, TimeSeriesSequence, WindowIndexer, WindowSpec, get_data
from tfts.features import FeatureDType, FeaturePipeline, FeatureRole, FeatureSpec, TimeSeriesSchema
from tfts.models.tft import TFTransformer, TFTransformerConfig
from tfts.training import TrainingArguments

# Smoke test pinning a single-device strategy so it runs identically on CI and
# any multi-GPU host.
_SINGLE_DEVICE_ARGS = TrainingArguments(output_dir="./weights", strategy="default")


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

    def test_model_with_static_and_quantile_inputs(self):
        config = TFTransformerConfig(
            static_real_dim=2,
            encoder_real_dim=4,
            decoder_real_dim=2,
            static_categorical_cardinalities=[5, 7],
            temporal_categorical_cardinalities=[13, 2],
            hidden_size=16,
            num_attention_heads=4,
            output_size=3,
            quantiles=[0.1, 0.5, 0.9],
        )
        model = TFTransformer(predict_sequence_length=6, config=config)
        inputs = {
            "static_categorical": tf.zeros([3, 2], dtype=tf.int32),
            "static_real": tf.zeros([3, 2]),
            "encoder_categorical": tf.zeros([3, 24, 2], dtype=tf.int32),
            "encoder_real": tf.zeros([3, 24, 4]),
            "decoder_categorical": tf.zeros([3, 6, 2], dtype=tf.int32),
            "decoder_real": tf.zeros([3, 6, 2]),
        }
        output = model(inputs)
        self.assertEqual(output.shape, (3, 6, 3))
        self.assertEqual(model.last_attention_weights.shape, (3, 6, 30))
        self.assertEqual(model.last_selection_weights["static"].shape, (3, 4))

    def test_canonical_batch_keeps_targets_and_all_covariate_types(self):
        config = TFTransformerConfig(
            encoder_real_dim=3,
            decoder_real_dim=2,
            static_real_dim=1,
            static_categorical_cardinalities=[5],
            temporal_categorical_cardinalities=[7, 3],
            hidden_size=16,
            num_attention_heads=4,
        )
        model = AutoModel.from_config(config, prediction_length=4)
        batch = TimeSeriesBatch(
            past_values=tf.random.normal([2, 8, 1]),
            past_time_features=tf.random.normal([2, 8, 2]),
            future_time_features=tf.random.normal([2, 4, 2]),
            past_categorical_features=tf.zeros([2, 8, 2], tf.int32),
            future_categorical_features=tf.zeros([2, 4, 2], tf.int32),
            static_real_features=tf.zeros([2, 1]),
            static_categorical_features=tf.zeros([2, 1], tf.int32),
        )

        output = model(batch)

        self.assertEqual(output.shape, (2, 4, 1))
        self.assertEqual(model.backbone.last_selection_weights["encoder"].shape, (2, 8, 5))
        self.assertEqual(model.backbone.last_selection_weights["decoder"].shape, (2, 4, 4))

    def test_canonical_batch_rejects_mismatched_tft_config(self):
        config = TFTransformerConfig(encoder_real_dim=2)
        model = AutoModel.from_config(config, prediction_length=4)
        batch = TimeSeriesBatch(
            past_values=tf.zeros([1, 8, 1]),
            past_time_features=tf.zeros([1, 8, 2]),
        )

        with self.assertRaisesRegex(ValueError, "TFT config expects 2"):
            model(batch)

    def test_materialized_mixed_categorical_roles_use_asymmetric_embeddings(self):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=12),
                "target": np.arange(12, dtype=np.float32),
                "observed_cat": np.arange(12) % 2,
                "known_cat": np.arange(12) % 3,
            }
        )
        schema = TimeSeriesSchema(
            "time",
            ("target",),
            (
                FeatureSpec("observed_cat", FeatureRole.OBSERVED_PAST, FeatureDType.CATEGORICAL),
                FeatureSpec("known_cat", FeatureRole.KNOWN_FUTURE, FeatureDType.CATEGORICAL),
            ),
        )
        prepared = FeaturePipeline().fit_transform(frame, schema)
        windows = WindowIndexer().build(prepared, WindowSpec(4, 2))
        batch = SequenceMaterializer().materialize(prepared, windows)
        config = TFTransformerConfig(
            encoder_real_dim=1,
            decoder_real_dim=1,
            encoder_categorical_cardinalities=[2, 3],
            decoder_categorical_cardinalities=[3],
            hidden_size=16,
            num_attention_heads=4,
        )

        output = AutoModel.from_config(config, prediction_length=2)(batch)

        self.assertEqual(output.shape, (len(windows), 2, 1))

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
        trainer = KerasTrainer(model, args=_SINGLE_DEVICE_ARGS)
        trainer.train(ts_sequence, epochs=1)


if __name__ == "__main__":
    unittest.main()
