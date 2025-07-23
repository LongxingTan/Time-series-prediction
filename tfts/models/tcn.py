"""
`WaveNet: A Generative Model for Raw Audio
<https://arxiv.org/abs/1609.03499>`_
"""

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Dropout, Lambda, ReLU, Reshape

from tfts.layers.cnn_layer import ConvTemp
from tfts.layers.dense_layer import DenseTemp

from .base import BaseConfig, BaseModel


class TCNConfig(BaseConfig):
    model_type: str = "tcn"

    def __init__(
        self,
        dilation_rates: List[int] = [2**i for i in range(4)],
        kernel_sizes: List[int] = [2 for _ in range(4)],
        filters: int = 128,
        dense_hidden_size: int = 64,
    ):
        """
        Initializes the configuration for the Temporal Convolutional Network (TCN) model with the specified parameters.

        Args:
            dilation_rates: List of dilation rates for each layer.
            kernel_sizes: List of kernel sizes for each convolutional layer.
            filters: The number of filters (channels) in each convolutional layer.
            dense_hidden_size: The size of the dense hidden layer.
        """
        super().__init__()
        self.dilation_rates: List[int] = dilation_rates
        self.kernel_sizes: List[int] = kernel_sizes
        self.filters: int = filters
        self.dense_hidden_size: int = dense_hidden_size


class TCN(BaseModel):
    """Temporal convolutional network"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[TCNConfig] = None) -> None:
        super(TCN, self).__init__()
        self.config = config or TCNConfig()
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(
            kernel_sizes=self.config.kernel_sizes,
            dilation_rates=self.config.dilation_rates,
            filters=self.config.filters,
            dense_hidden_size=self.config.dense_hidden_size,
        )

        self.project1 = Dense(predict_sequence_length, activation=None)

        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation="relu")

        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation="relu")

    def __call__(
        self,
        inputs: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """TCN call

        Parameters
        ----------
        inputs : tf.Tensor
            3D input tensor
        teacher : tf.Tensor, optional
            _description_, by default None

        Returns
        -------
        tf.Tensor
            3D output tensor
        """
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs)

        # encoder_features = self.pool(encoder_features)  # batch * n_train_days * n_feature
        encoder_outputs, encoder_state = self.encoder(encoder_feature)
        # outputs = self.dense1(encoder_state)  # batch * predict_sequence_length
        # outputs = self.dense2(encoder_outputs)[:, -self.predict_sequence_length]

        if output_hidden_states:
            return encoder_outputs

        memory = encoder_state[:, -1]
        encoder_output = self.drop1(memory)
        encoder_output = self.dense1(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = Reshape((outputs.shape[1], 1))(outputs)

        # outputs = tf.tile(outputs, (1, self.predict_sequence_length, 1))   # stupid
        # outputs = self.dense3(encoder_outputs)
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, kernel_sizes, dilation_rates, filters, dense_hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        self.filters = filters
        self.dense_hidden_size = dense_hidden_size
        self.conv_times = []

    def build(self, input_shape):
        super(Encoder, self).build(input_shape)
        _, time_steps, input_dim = input_shape

        self.dense_time1 = DenseTemp(hidden_size=self.filters, activation="tanh", name="encoder_dense_time1")
        self.dense_time1.build((None, time_steps, input_dim))

        conv_input_shape = (None, time_steps, self.filters)
        for i, (kernel_size, dilation) in enumerate(zip(self.kernel_sizes, self.dilation_rates)):
            conv_temp = ConvTemp(filters=2 * self.filters, kernel_size=kernel_size, causal=True, dilation_rate=dilation)
            conv_temp.build(conv_input_shape)
            self.conv_times.append(conv_temp)

        self.dense_time2 = DenseTemp(hidden_size=self.filters + self.filters, name="encoder_dense_time2")
        self.dense_time2.build((None, time_steps, self.filters))
        self.dense_time3 = DenseTemp(hidden_size=self.dense_hidden_size, activation="relu", name="encoder_dense_time3")
        self.dense_time3.build((None, time_steps, self.filters))
        self.dense_time4 = DenseTemp(hidden_size=1, name="encoder_dense_time_4")
        self.dense_time4.build((None, time_steps, self.dense_hidden_size))
        self.built = True

    def call(self, x: tf.Tensor):
        # => batch_size * time_sequence_length * filters
        inputs = self.dense_time1(inputs=x)

        skip_outputs = []
        conv_inputs = [inputs]
        for conv_time in self.conv_times:
            dilated_conv = conv_time(inputs)

            split_layer = Lambda(lambda x: tf.split(x, 2, axis=2))
            conv_filter, conv_gate = split_layer(dilated_conv)
            # dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
            dilated_conv = Lambda(lambda x: tf.nn.tanh(x[0]) * tf.nn.sigmoid(x[1]))([conv_filter, conv_gate])
            outputs = self.dense_time2(inputs=dilated_conv)
            # skips, residuals = tf.split(outputs, [self.filters, self.filters], axis=2)
            split_layer2 = Lambda(lambda x: tf.split(x, [self.filters, self.filters], axis=2))
            skips, residuals = split_layer2(outputs)
            inputs += residuals
            conv_inputs.append(inputs)  # batch_size * time_sequence_length * filters
            skip_outputs.append(skips)

        # skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        concat_layer = Concatenate(axis=2)
        concatenated = concat_layer(skip_outputs)
        relu_layer = ReLU()
        skip_outputs = relu_layer(concatenated)
        h = self.dense_time3(skip_outputs)
        # y_hat = self.dense_time4(h)
        return conv_inputs[:-1], h

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update(
            {
                "kernel_sizes": self.kernel_sizes,
                "dilation_rates": self.dilation_rates,
                "filters": self.filters,
                "dense_hidden_size": self.dense_hidden_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0]
        time_sequence_length = input_shape[1]

        # After dense_time1: (batch_size, time_sequence_length, filters)
        intermediate_shape = (batch_size, time_sequence_length, self.filters)

        # conv_inputs contains intermediate representations after each conv layer
        # conv_inputs[:-1] excludes the last element, so we have len(self.conv_times) shapes
        conv_inputs_shapes = [intermediate_shape] * len(self.conv_times)

        # After dense_time3: (batch_size, time_sequence_length, dense_hidden_size)
        h_shape = (batch_size, time_sequence_length, self.dense_hidden_size)

        return (conv_inputs_shapes, h_shape)
