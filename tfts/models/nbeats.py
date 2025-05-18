"""
`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
<https://arxiv.org/abs/1905.10437>`_
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Lambda

from tfts.layers.nbeats_layer import GenericBlock, SeasonalityBlock, TrendBlock

from .base import BaseConfig, BaseModel


class NBeatsConfig(BaseConfig):
    model_type: str = "nbeats"

    def __init__(
        self,
        stack_types=["trend_block", "seasonality_block"],
        nb_blocks_per_stack=3,
        n_block_layers=4,
        hidden_size=64,
        thetas_dims=(4, 8),
        share_weights_in_stack=False,
    ):
        super(NBeatsConfig, self).__init__()
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.n_block_layers = n_block_layers
        self.hidden_size = hidden_size
        self.thetas_dims = thetas_dims
        self.share_weights_in_stack = share_weights_in_stack


class NBeats(BaseModel):
    """NBeats model"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        config: Optional[NBeatsConfig] = None,
    ):
        super(NBeats, self).__init__(config)
        self.config = config or NBeatsConfig()
        self.predict_sequence_length = predict_sequence_length

        self.stack_types = self.config.stack_types
        self.nb_blocks_per_stack = self.config.nb_blocks_per_stack
        self.hidden_size = self.config.hidden_size
        self.n_block_layers = self.config.n_block_layers

        self.block_type = {"trend_block": TrendBlock, "seasonality_block": SeasonalityBlock, "general": GenericBlock}

    def __call__(
        self, inputs: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        if isinstance(inputs, (list, tuple)):
            print("NBeats only support single variable prediction, so ignore encoder_features and decoder_features")
            x, encoder_features, _ = inputs
        else:  # for single variable prediction
            x = inputs

        shape_fn = Lambda(lambda t: tf.shape(t)[1])
        squeeze_fn = Lambda(lambda t: tf.squeeze(t, 2))
        self.train_sequence_length = shape_fn(x)
        x = squeeze_fn(x)

        self.stacks = []
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

        forecast = tf.zeros([tf.shape(x)[0], self.predict_sequence_length], dtype=tf.float32)
        backcast = x

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = Lambda(lambda x: x[0] - x[1])([backcast, b])
                forecast = Lambda(lambda x: x[0] + x[1])([forecast, f])

        # Final expansion using Keras layer
        forecast = Lambda(lambda x: tf.expand_dims(x, -1))(forecast)
        return forecast

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = self.block_type[stack_type]

            block = block_init(
                self.train_sequence_length, self.predict_sequence_length, self.hidden_size, self.n_block_layers
            )
            blocks.append(block)
        return blocks
