"""Test the TFT model"""

import unittest

import tensorflow as tf

from tfts.models.tft import TFTransformer, TFTransformerConfig


class TFTransformerTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        custom_model_config = TFTransformerConfig(
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

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")


if __name__ == "__main__":
    unittest.main()
