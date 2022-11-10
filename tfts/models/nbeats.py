"""
`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
<https://arxiv.org/abs/1905.10437>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf

from tfts.layers.nbeats_layer import GenericBlock, SeasonalityBlock, TrendBlock

params = {
    "stack_types": ["trend_block", "seasonality_block"],
    "nb_blocks_per_stack": 3,
    "n_block_layers": 4,
    "hidden_size": 64,
    "thetas_dims": (4, 8),
    "share_weights_in_stack": False,
}


class NBeats(object):
    """NBeats model"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        custom_model_params: Optional[Dict[str, Any]] = None,
        custom_model_head: Optional[Callable] = None,
    ):
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length

        self.stack_types = params["stack_types"]
        self.nb_blocks_per_stack = params["nb_blocks_per_stack"]
        self.hidden_size = params["hidden_size"]
        self.n_block_layers = params["n_block_layers"]

        self.block_type = {"trend_block": TrendBlock, "seasonality_block": SeasonalityBlock, "general": GenericBlock}

    def __call__(self, inputs):
        if isinstance(inputs, (list, tuple)):
            print("NBeats only support single variable prediction, so ignore encoder_features and decoder_features")
            x, encoder_features, _ = inputs
        else:  # for single variable prediction
            x = inputs

        x = tf.squeeze(x, 2)  # 3 dim for all models
        # Todo: if train_length and predict_length is both 12, train fail
        self.train_sequence_length = x.get_shape().as_list()[1]

        self.stacks = []
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

        forecast = tf.zeros([tf.shape(x)[0], self.predict_sequence_length], dtype=tf.float32)
        backcast = x
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        forecast = tf.expand_dims(forecast, -1)
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
