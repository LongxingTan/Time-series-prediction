"""
`Long short term memory
<http://www.bioinf.jku.at/publications/older/2604.pdf>`_
"""

from typing import Dict, Literal, Optional

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, AveragePooling1D, Bidirectional, Concatenate, Dense, Reshape

from .base import BaseConfig, BaseModel


class RNNConfig(BaseConfig):
    model_type: str = "rnn"

    def __init__(
        self,
        rnn_hidden_size: int = 64,
        rnn_type: Literal["gru", "lstm"] = "gru",
        bi_direction: bool = False,
        dense_hidden_size: int = 128,
        num_stacked_layers: int = 1,
        scheduled_sampling: float = 0.0,
        use_attention: bool = False,
    ) -> None:
        """
        Initializes the configuration for the RNN model with the specified parameters.

        Args:
            rnn_hidden_size: The number of units in the RNN hidden layer.
            rnn_type: Type of RNN ('gru' or 'lstm').
            bi_direction: Whether to use bidirectional RNN.
            dense_hidden_size: The size of the dense hidden layer following the RNN.
            num_stacked_layers: The number of stacked RNN layers.
            scheduled_sampling: Scheduled sampling ratio.
            use_attention: Whether to use attention mechanism.
        """
        super().__init__()

        self.rnn_hidden_size: int = rnn_hidden_size
        self.rnn_type: Literal["gru", "lstm"] = rnn_type
        self.bi_direction: bool = bi_direction
        self.dense_hidden_size: int = dense_hidden_size
        self.num_stacked_layers: int = num_stacked_layers
        self.scheduled_sampling: float = scheduled_sampling
        self.use_attention: bool = use_attention


class RNN(BaseModel):
    """tfts RNN model"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(rnn_size=self.config.rnn_hidden_size, rnn_type=self.config.rnn_type, return_state=True)

        self.dense1 = Dense(self.config.dense_hidden_size, activation="relu")
        self.dense2 = Dense(self.config.dense_hidden_size, activation="relu")
        self.project1 = Dense(predict_sequence_length, activation=None)

    def __call__(
        self, inputs, teacher=None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        """RNN model call

        Parameters
        ----------
        inputs : Union[list, tuple, dict, tf.Tensor]
            Input data.
        teacher : tf.Tensor, optional
            Teacher signal for training, by default None.
        return_dict : bool, optional
            Whether to return a dictionary, by default None.

        Returns
        -------
        tf.Tensor
            Model output.
        """
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs)
        encoder_outputs, encoder_state = self.encoder(encoder_feature)

        encoder_out = self.dense1(encoder_state)
        encoder_out = self.dense2(encoder_out)
        outputs = self.project1(encoder_out)
        outputs = Reshape((outputs.shape[1], 1))(outputs)
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_size: int,
        rnn_type: str = "gru",
        rnn_dropout: float = 0,
        num_stacked_layers: int = 1,
        bi_direction: bool = False,
        return_state: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in ("gru", "lstm"):
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")

        self.rnn_size = rnn_size
        self.rnn_dropout = rnn_dropout
        self.num_stacked_layers = num_stacked_layers
        self.bi_direction = bi_direction
        self.return_state = return_state
        self.rnn_layers = []

    def build(self, input_shape):
        for i in range(self.num_stacked_layers):
            return_state = self.return_state if i == self.num_stacked_layers - 1 and not self.bi_direction else False

            rnn_class = GRU if self.rnn_type == "gru" else LSTM
            rnn = rnn_class(
                units=self.rnn_size,
                activation="tanh",
                return_sequences=True,
                return_state=return_state,
                dropout=self.rnn_dropout if self.rnn_dropout > 0 else 0.0,
            )

            if self.bi_direction:
                rnn = Bidirectional(rnn)

            self.rnn_layers.append(rnn)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor):
        """RNN encoder call

        Parameters
        ----------
        inputs : tf.Tensor
            3d time series input, (batch_size, sequence_length, num_features)

        Returns
        -------
        tf.Tensor or tuple
            If return_state is False:
                output tensor: (batch_size, sequence_length, rnn_size)
            If return_state is True:
                For GRU: (output, state)
                    output: (batch_size, sequence_length, rnn_size)
                    state: (batch_size, rnn_size)
                For LSTM: (output, state_h, state_c)
                    output: (batch_size, sequence_length, rnn_size)
                    state_h: (batch_size, rnn_size)
                    state_c: (batch_size, rnn_size)
        """
        x = inputs

        for i, layer in enumerate(self.rnn_layers):
            is_last_layer = i == len(self.rnn_layers) - 1
            if is_last_layer and self.return_state:
                outputs = layer(x)

                if self.bi_direction:
                    output = outputs[0]
                    if self.rnn_type == "gru":
                        # GRU: forward_state, backward_state
                        fw_state, bw_state = outputs[1:]
                        state = tf.concat([fw_state, bw_state], axis=-1)
                        return output, state
                    else:  # LSTM
                        fw_h, fw_c, bw_h, bw_c = outputs[1:]
                        state_h = tf.concat([fw_h, bw_h], axis=-1)
                        state_c = tf.concat([fw_c, bw_c], axis=-1)
                        state = Concatenate(axis=-1)([state_h, state_c])
                        return output, state
                else:
                    if self.rnn_type == "lstm":
                        x, state_h, state_c = outputs
                        state = Concatenate(axis=-1)([state_h, state_c])
                    else:
                        x, state = outputs
                    return x, state
            else:
                x = layer(x)
        return x

    def get_config(self):
        config = {
            "rnn_type": self.rnn_type,
            "rnn_size": self.rnn_size,
            "rnn_dropout": self.rnn_dropout,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the encoder."""
        batch_size, seq_length, _ = input_shape

        if self.bi_direction:
            rnn_output_size = 2 * self.rnn_size
        else:
            rnn_output_size = self.rnn_size

        if not self.return_state:
            return (batch_size, seq_length, rnn_output_size)

        else:
            if self.bi_direction:
                # Bidirectional with return_state is always just the output sequence
                return (batch_size, seq_length, rnn_output_size)
            elif self.rnn_type == "gru":
                # GRU: (output, state)
                return ((batch_size, seq_length, rnn_output_size), (batch_size, rnn_output_size))
            else:
                # LSTM: (output, state_h, state_c)
                return ((batch_size, seq_length, rnn_output_size), (batch_size, 2 * rnn_output_size))


class RNN2(object):
    def __init__(self, predict_sequence_length=3, config=RNNConfig()) -> None:
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.rnn = GRU(
            units=config.rnn_hidden_size, activation="tanh", return_state=True, return_sequences=True, dropout=0
        )
        self.dense1 = Dense(predict_sequence_length)
        self.dense2 = Dense(1)

    def __call__(self, inputs: tf.Tensor, teacher: Optional[tf.Tensor] = None):
        """Another RNN model

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
            x, encoder_features, _ = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs

        encoder_shape = tf.shape(encoder_features)
        future = tf.zeros([encoder_shape[0], self.predict_sequence_length, encoder_shape[2]])
        encoder_features = tf.concat([encoder_features, future], axis=1)
        output, state = self.rnn(encoder_features)
        output = self.dense2(output)

        return output[:, -self.predict_sequence_length :]
