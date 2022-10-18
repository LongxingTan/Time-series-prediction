"""
`N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
<https://arxiv.org/abs/1905.10437>`_
"""

from collections import defaultdict

import tensorflow as tf

from tfts.layers.nbeats_layer import GenericBlock, SeasonalityBlock, TrendBlock

params = defaultdict(
    stack_types=["trend_block", "seasonality_block"],
    nb_blocks_per_stack=3,
    hidden_layer_units=256,
    thetas_dims=(4, 8),
    share_weights_in_stack=False,
)


class NBeats(object):
    """NBeats model"""

    def __init__(self, predict_sequence_length, custom_model_params=None):
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.backcast_length = predict_sequence_length

        self.stack_types = params["stack_types"]
        self.nb_blocks_per_stack = params["nb_blocks_per_stack"]
        self.hidden_layer_units = params["hidden_layer_units"]
        self.theta_dims = params["thetas_dims"]
        self.share_weights_in_stack = params["share_weights_in_stack"]

        self.block_type = {"trend_block": TrendBlock, "seasonality_block": SeasonalityBlock, "general": GenericBlock}

    def __call__(self, x):
        self.forecast_length = x.get_shape().as_list()[1]

        self.stacks = []
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

        forecast = tf.zeros([tf.shape(x)[0], self.forecast_length], dtype=tf.float32)
        backcast = x
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        return forecast

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = self.block_type[stack_type]
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]
            else:
                block = block_init(
                    self.hidden_layer_units, self.theta_dims[stack_id], self.backcast_length, self.forecast_length
                )
            blocks.append(block)
        return blocks
