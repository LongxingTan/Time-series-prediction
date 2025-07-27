"""
`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
<https://arxiv.org/abs/1905.10437>`_
"""

from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Add, Lambda, Layer, Subtract

from ..layers.nbeats_layer import GenericBlock, SeasonalityBlock, TrendBlock
from ..layers.util_layer import ShapeLayer, ZerosLayer
from .base import BaseConfig, BaseModel


class NBeatsConfig(BaseConfig):
    model_type: str = "nbeats"

    def __init__(
        self,
        stack_types=["trend_block", "seasonality_block"],
        nb_blocks_per_stack=3,
        num_block_layers=4,
        hidden_size=64,
        thetas_dims=(4, 8),
        share_weights_in_stack=False,
    ):
        super(NBeatsConfig, self).__init__()
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.num_block_layers = num_block_layers
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
        super().__init__()
        self.config = config or NBeatsConfig()
        self.predict_sequence_length = predict_sequence_length
        self.train_sequence_length = 1  # Temp

        self.stack_types = self.config.stack_types
        self.nb_blocks_per_stack = self.config.nb_blocks_per_stack
        self.hidden_size = self.config.hidden_size
        self.num_block_layers = self.config.num_block_layers

        # Create custom layers
        self.shape_layer = ShapeLayer()
        self.squeeze_layer = Lambda(lambda t: tf.squeeze(t, 2))
        self.zeros_layer = ZerosLayer(predict_sequence_length)
        self.expand_dims_layer = Lambda(lambda x: tf.expand_dims(x, -1))

        self.block_type = {"trend_block": TrendBlock, "seasonality_block": SeasonalityBlock, "general": GenericBlock}
        self.stacks = []
        for stack_type in self.stack_types:
            self.stacks.append(self.create_stack(stack_type))

    def __call__(
        self, inputs: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        if isinstance(inputs, (list, tuple)):
            print("NBeats only support single variable prediction, so ignore encoder_features and decoder_features")
            x, encoder_features, _ = inputs
        else:  # for single variable prediction
            x = inputs

        shape = self.shape_layer(x)
        self.train_sequence_length = shape[1]
        x = self.squeeze_layer(x)

        forecast = self.zeros_layer(x)
        backcast = x

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = Subtract()([backcast, b])
                forecast = Add()([forecast, f])

        # Final expansion using Keras layer
        forecast = self.expand_dims_layer(forecast)
        return forecast

    def create_stack(self, stack_type):
        blocks: List[Layer] = []
        for block_id in range(self.nb_blocks_per_stack):
            block_fn = self.block_type[stack_type]
            block = block_fn(
                self.train_sequence_length, self.predict_sequence_length, self.hidden_size, self.num_block_layers
            )
            blocks.append(block)
        return blocks
