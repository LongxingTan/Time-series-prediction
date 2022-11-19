"""
`WaveNet: A Generative Model for Raw Audio
<https://arxiv.org/abs/1609.03499>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten

from tfts.layers.cnn_layer import ConvTemp
from tfts.layers.dense_layer import DenseTemp

params = {
    "dilation_rates": [2**i for i in range(4)],
    "kernel_sizes": [2 for i in range(4)],
    "filters": 128,
    "dense_hidden_size": 64,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class TCN(object):
    """Temporal convolutional network"""

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
        self.encoder = Encoder(
            params["kernel_sizes"], params["dilation_rates"], params["filters"], params["dense_hidden_size"]
        )
        # self.dense2 = Dense(1)
        # self.dense3 = TimeDistributed(Dense(1))
        # self.pool = AveragePooling1D(pool_size=144, strides=144, padding='valid')

        self.project1 = Dense(predict_sequence_length, activation=None)
        # self.project1 = Dense(48, activation=None)

        # self.bn1 = BatchNormalization()
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation="relu")

        # self.bn2 = BatchNormalization()
        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation="relu")

    def __call__(self, inputs, teacher=None):
        """_summary_

        Parameters
        ----------
        inputs : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs

        # encoder_features = self.pool(encoder_features)  # batch * n_train_days * n_feature
        encoder_outputs, encoder_state = self.encoder(encoder_feature)
        # outputs = self.dense1(encoder_state)  # batch * predict_sequence_length
        # outputs = self.dense2(encoder_outputs)[:, -self.predict_sequence_length]
        # print(len(encoder_outputs), encoder_outputs[0].shape, encoder_state.shape)

        memory = encoder_state[:, -1]
        encoder_output = self.drop1(memory)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.bn2(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)

        # outputs = tf.tile(outputs, (1, self.predict_sequence_length, 1))   # stupid
        # outputs = self.dense3(encoder_outputs)

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, :]
            outputs = outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x, axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            outputs = outputs + x_mean
        return outputs


class Encoder(object):
    def __init__(self, kernel_sizes, dilation_rates, filters, dense_hidden_size):
        self.filters = filters
        self.conv_times = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilation_rates)):
            self.conv_times.append(
                ConvTemp(filters=2 * filters, kernel_size=kernel_size, causal=True, dilation_rate=dilation)
            )
        self.dense_time1 = DenseTemp(hidden_size=filters, activation="tanh", name="encoder_dense_time1")
        self.dense_time2 = DenseTemp(hidden_size=filters + filters, name="encoder_dense_time2")
        self.dense_time3 = DenseTemp(hidden_size=dense_hidden_size, activation="relu", name="encoder_dense_time3")
        self.dense_time4 = DenseTemp(hidden_size=1, name="encoder_dense_time_4")

    def __call__(self, x):
        inputs = self.dense_time1(inputs=x)  # batch_size * time_sequence_length * filters

        skip_outputs = []
        conv_inputs = [inputs]
        for conv_time in self.conv_times:
            dilated_conv = conv_time(inputs)
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
            outputs = self.dense_time2(inputs=dilated_conv)
            skips, residuals = tf.split(outputs, [self.filters, self.filters], axis=2)
            inputs += residuals
            conv_inputs.append(inputs)  # batch_size * time_sequence_length * filters
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = self.dense_time3(skip_outputs)
        # y_hat = self.dense_time4(h)
        return conv_inputs[:-1], h
