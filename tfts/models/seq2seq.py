"""
`Sequence to Sequence Learning with Neural Networks
<https://arxiv.org/abs/1409.3215>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, RNN, Dense, Dropout, GRUCell, LSTMCell

from tfts.layers.attention_layer import FullAttention

params = {
    "rnn_type": "gru",
    "bi_direction": False,
    "rnn_size": 64,
    "dense_size": 64,
    "num_stacked_layers": 1,
    "scheduler_sampling": 0,  # teacher forcing
    "use_attention": False,
    "attention_sizes": 64,
    "attention_heads": 2,
    "attention_dropout": 0,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class Seq2seq(object):
    """Seq2seq model"""

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
            rnn_type=params["rnn_type"], rnn_size=params["rnn_size"], dense_size=params["dense_size"]
        )
        self.decoder = Decoder1(
            rnn_type=params["rnn_type"],
            rnn_size=params["rnn_size"],
            predict_sequence_length=predict_sequence_length,
            use_attention=params["use_attention"],
            attention_sizes=params["attention_sizes"],
            attention_heads=params["attention_heads"],
            attention_dropout=params["attention_dropout"],
        )

    def __call__(self, inputs, teacher=None):
        """An RNN seq2seq structure for time series

        :param inputs: _description_
        :type inputs: _type_
        :param teacher: teacher forcing decoding, defaults to None
        :type teacher: _type_, optional
        :return: _description_
        :type: _type_
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            decoder_feature = inputs["decoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs
            decoder_feature = tf.cast(
                tf.tile(
                    tf.reshape(tf.range(self.predict_sequence_length), (1, self.predict_sequence_length, 1)),
                    (tf.shape(encoder_feature)[0], 1, 1),
                ),
                tf.float32,
            )

        encoder_outputs, encoder_state = self.encoder(encoder_feature)

        decoder_outputs = self.decoder(
            decoder_feature,
            decoder_init_input=x[:, -1, 0:1],
            init_state=encoder_state,
            teacher=teacher,
            scheduler_sampling=self.params["scheduler_sampling"],
            encoder_output=encoder_outputs,
        )

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            decoder_outputs = decoder_outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            decoder_outputs = decoder_outputs + x_mean
        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_type, rnn_size, rnn_dropout=0, dense_size=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn_type = rnn_type
        if rnn_type.lower() == "gru":
            self.rnn = GRU(
                units=rnn_size, activation="tanh", return_state=True, return_sequences=True, dropout=rnn_dropout
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                units=rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=rnn_dropout,
            )
        self.dense = Dense(units=dense_size, activation="tanh")

    def call(self, inputs):
        """Seq2seq encoder

        Parameters
        ----------
        inputs : tf.Tensor
            _description_

        Returns
        -------
        tf.Tensor
            batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        """
        if self.rnn_type.lower() == "gru":
            outputs, state = self.rnn(inputs)
            state = self.dense(state)
        elif self.rnn_type.lower() == "lstm":
            outputs, state1, state2 = self.rnn(inputs)
            state = (state1, state2)
        else:
            raise ValueError("No supported rnn type of {}".format(self.rnn_type))
        # encoder_hidden_state = tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        # outputs = self.dense(outputs)  # => batch_size * input_seq_length * dense_size
        return outputs, state


class Decoder1(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_type="gru",
        rnn_size=32,
        predict_sequence_length=3,
        use_attention=False,
        attention_sizes=32,
        attention_heads=1,
        attention_dropout=0.0,
    ):
        super(Decoder1, self).__init__()
        self.predict_sequence_length = predict_sequence_length
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.attention_sizes = attention_sizes
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        if self.rnn_type.lower() == "gru":
            self.rnn_cell = GRUCell(self.rnn_size)
        elif self.rnn_type.lower() == "lstm":
            self.rnn_cell = LSTMCell(units=self.rnn_size)
        self.dense = Dense(units=1, activation=None)
        if self.use_attention:
            self.attention = FullAttention(
                hidden_size=self.attention_sizes,
                num_heads=self.attention_heads,
                attention_dropout=self.attention_dropout,
            )
        super().build(input_shape)

    def call(
        self,
        decoder_features,
        decoder_init_input,
        init_state,
        teacher=None,
        scheduler_sampling=0,
        training=None,
        **kwargs
    ):
        """Seq2seq decoder1: step by step

        :param decoder_features: _description_
        :type decoder_features: _type_
        :param decoder_init_input: _description_
        :type decoder_init_input: _type_
        :param init_state: _description_
        :type init_state: _type_
        :param teacher: _description_, defaults to None
        :type teacher: _type_, optional
        :param scheduler_sampling: _description_, defaults to 0
        :type scheduler_sampling: int, optional
        :param training: _description_, defaults to None
        :type training: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        decoder_outputs = []
        prev_output = decoder_init_input
        prev_state = init_state
        if teacher is not None:
            teacher = tf.squeeze(teacher, 2)
            teachers = tf.split(teacher, self.predict_sequence_length, axis=1)

        for i in range(self.predict_sequence_length):
            if training:
                p = np.random.uniform(low=0, high=1, size=1)[0]
                if teacher is not None and p > scheduler_sampling:
                    this_input = teachers[i]
                else:
                    this_input = prev_output
            else:
                this_input = prev_output

            if decoder_features is not None:
                this_input = tf.concat([this_input, decoder_features[:, i]], axis=-1)

            if self.use_attention:
                if self.rnn_type.lower() == "gru":
                    # q: (batch, 1, feature), att_output: (batch, 1, feature)
                    att = self.attention(
                        tf.expand_dims(prev_state, 1), k=kwargs["encoder_output"], v=kwargs["encoder_output"]
                    )
                    att = tf.squeeze(att, 1)  # (batch, feature)
                elif self.rnn_type.lower() == "lstm":
                    # q: (batch, 1, feature * 2), att_output: (batch, 1, feature)
                    att = self.attention(
                        tf.expand_dims(tf.concat(prev_state, 1), 1),
                        k=kwargs["encoder_output"],
                        v=kwargs["encoder_output"],
                    )
                    att = tf.squeeze(att, 1)  # (batch, feature)

                this_input = tf.concat([this_input, att], axis=-1)

            this_output, this_state = self.rnn_cell(this_input, prev_state)
            prev_state = this_state
            prev_output = self.dense(this_output)
            decoder_outputs.append(prev_output)

        decoder_outputs = tf.concat(decoder_outputs, axis=-1)
        return tf.expand_dims(decoder_outputs, -1)


class Decoder2(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_type="gru",
        rnn_size=32,
        predict_sequence_length=3,
        use_attention=False,
        attention_sizes=32,
        attention_heads=1,
        attention_dropout=0.0,
    ):
        super(Decoder2, self).__init__()
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.predict_sequence_length = predict_sequence_length
        self.use_attention = use_attention
        self.attention_sizes = attention_sizes
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        if self.rnn_type.lower() == "gru":
            self.rnn_cell = GRUCell(self.rnn_size)
        elif self.rnn_type.lower() == "lstm":
            self.rnn = LSTMCell(units=self.rnn_size)
        self.dense = Dense(units=1)
        if self.use_attention:
            self.attention = FullAttention(
                hidden_size=self.attention_sizes,
                num_heads=self.attention_heads,
                attention_dropout=self.attention_dropout,
            )
        super().build(input_shape)

    def forward(
        self,
        decoder_feature,
        decoder_init_value,
        init_state,
        teacher=None,
        scheduler_sampling=0,
        training=None,
        **kwargs
    ):
        def cond_fn(time, prev_output, prev_state, decoder_output_ta):
            return time < self.predict_sequence_length

        def body(time, prev_output, prev_state, decoder_output_ta):
            if time == 0 or teacher is None:
                this_input = prev_output

            else:
                this_input = teacher[:, time - 1, :]

            if decoder_feature is not None:
                this_feature = decoder_feature[:, time, :]
                this_input = tf.concat([this_input, this_feature], axis=1)

            if self.use_attention:
                attention_feature = self.attention(
                    tf.expand_dims(prev_state[-1], 1), k=kwargs["encoder_output"], v=kwargs["encoder_output"]
                )
                attention_feature = tf.squeeze(attention_feature, 1)
                this_input = tf.concat([this_input, attention_feature], axis=-1)

            this_output, this_state = self.rnn_cell(this_input, prev_state)
            project_output = self.dense(this_output)
            decoder_output_ta = decoder_output_ta.write(time, project_output)
            return time + 1, project_output, this_state, decoder_output_ta

        loop_init = [
            tf.constant(0, dtype=tf.int32),  # steps
            decoder_init_value,  # decoder each step
            init_state,  # state
            tf.TensorArray(dtype=tf.float32, size=self.predict_sequence_length),
        ]
        _, _, _, decoder_outputs_ta = tf.while_loop(cond_fn, body, loop_init)

        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs

    def call(
        self,
        decoder_feature,
        decoder_init_input,
        init_state,
        teacher=None,
        scheduler_sampling=0,
        training=None,
        **kwargs
    ):
        """Decoder model2

        Parameters
        ----------
        decoder_feature : _type_
            _description_
        init_state : _type_
            _description_
        decoder_init_input : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        return self.forward(
            decoder_feature=decoder_feature,
            decoder_init_value=decoder_init_input,
            init_state=[init_state],  # for tf2
            teacher=teacher,
        )


class Decoder3(tf.keras.layers.Layer):
    # multi-steps static decoding
    def __init__(self, rnn_type="gru", rnn_size=32, rnn_dropout=0, dense_size=1, **kwargs) -> None:
        super(Decoder3, self).__init__()
        if rnn_type.lower() == "gru":
            self.rnn = GRU(
                units=rnn_size, activation="tanh", return_state=False, return_sequences=True, dropout=rnn_dropout
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                units=rnn_size,
                activation="tanh",
                return_state=False,
                return_sequences=True,
                dropout=rnn_dropout,
            )
        self.dense = Dense(units=dense_size, activation=None)
        self.drop = Dropout(0.1)

    def call(
        self,
        decoder_features,
        decoder_init_input,
        init_state,
        teacher=None,
        scheduler_sampling=0,
        training=None,
        **kwargs
    ):
        """Decoder3: just simple

        Parameters
        ----------
        decoder_features : _type_
            _description_
        decoder_init_input : _type_
            _description_
        init_state : _type_
            _description_
        teacher : _type_, optional
            _description_, by default None
        scheduler_sampling : int, optional
            _description_, by default 0
        training : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        x = self.rnn(decoder_features, initial_state=init_state)
        # x = self.drop(x)
        x = self.dense(x)
        return x
