"""Stacked recurrent encoder used by the TFTS RNN model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Concatenate


class UnsupportedRecurrentCell(ValueError):
    """An encoder was requested with an unknown recurrent cell type."""

    def __init__(self, cell_type: str) -> None:
        self.cell_type = cell_type
        super().__init__(f"Unsupported RNN type: {cell_type}")


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
            raise UnsupportedRecurrentCell(self.rnn_type)

        self.rnn_size = rnn_size
        self.rnn_dropout = rnn_dropout
        self.num_stacked_layers = num_stacked_layers
        self.bi_direction = bi_direction
        self.return_state = return_state
        self.rnn_layers = []
        self.supports_masking = True

    def build(self, input_shape):
        layer_shape = tf.TensorShape(input_shape)
        for i in range(self.num_stacked_layers):
            return_state = self.return_state and i == self.num_stacked_layers - 1

            rnn_class = GRU if self.rnn_type == "gru" else LSTM
            rnn_kwargs = dict(
                units=self.rnn_size,
                activation="tanh",
                return_sequences=True,
                return_state=return_state,
                dropout=self.rnn_dropout if self.rnn_dropout > 0 else 0.0,
            )
            # ``reset_after`` is a GRU-only argument in current Keras.  Passing
            # it to LSTM worked in older releases but now raises at call time.
            if rnn_class is GRU:
                rnn_kwargs["reset_after"] = False
            rnn = rnn_class(**rnn_kwargs)

            if self.bi_direction:
                rnn = Bidirectional(rnn)

            rnn.build(layer_shape)
            self.rnn_layers.append(rnn)
            layer_shape = layer_shape[:-1].concatenate(self.rnn_size * (2 if self.bi_direction else 1))

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, mask=None, training: bool | None = None):
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
                Returns (sequence, state). The sequence width is rnn_size
                times the number of directions. GRU state has that width;
                LSTM state concatenates hidden and cell states and has twice
                that width.
        """
        x = inputs

        for i, layer in enumerate(self.rnn_layers):
            is_last_layer = i == len(self.rnn_layers) - 1
            if is_last_layer and self.return_state:
                outputs = layer(x, mask=mask, training=training)

                if self.bi_direction:
                    output = outputs[0]
                    if self.rnn_type == "gru":
                        # GRU: forward_state, backward_state
                        fw_state, bw_state = outputs[1:]
                        state = Concatenate(axis=-1)([fw_state, bw_state])
                        return output, state
                    else:  # LSTM
                        fw_h, fw_c, bw_h, bw_c = outputs[1:]
                        state_h = Concatenate(axis=-1)([fw_h, bw_h])
                        state_c = Concatenate(axis=-1)([fw_c, bw_c])
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
                x = layer(x, mask=mask, training=training)
        return x

    def get_config(self):
        config = {
            "rnn_type": self.rnn_type,
            "rnn_size": self.rnn_size,
            "rnn_dropout": self.rnn_dropout,
            "num_stacked_layers": self.num_stacked_layers,
            "bi_direction": self.bi_direction,
            "return_state": self.return_state,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        """Only the sequence output carries a temporal padding mask."""
        return (mask, None) if self.return_state else mask

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the encoder."""
        batch_size, seq_length, _ = input_shape

        if self.bi_direction:
            rnn_output_size = 2 * self.rnn_size
        else:
            rnn_output_size = self.rnn_size

        if not self.return_state:
            return (batch_size, seq_length, rnn_output_size)

        state_size = rnn_output_size * (2 if self.rnn_type == "lstm" else 1)
        return ((batch_size, seq_length, rnn_output_size), (batch_size, state_size))
