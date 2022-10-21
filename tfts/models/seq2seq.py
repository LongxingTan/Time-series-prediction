"""
`Sequence to Sequence Learning with Neural Networks
<https://arxiv.org/abs/1409.3215>`_
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, RNN, Dense, Dropout, GRUCell, LSTMCell

from tfts.layers.attention_layer import FullAttention

params = {
    "rnn_type": "gru",
    "bi_direction": False,
    "rnn_size": 64,
    "dense_size": 64,
    "num_stacked_layers": 1,
    "scheduler_sampling": 0,
    "use_attention": False,
    "attention_sizes": 64,
    "attention_heads": 2,
    "attention_dropout": 0,
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class Seq2seq(object):
    """Seq2seq model"""

    def __init__(self, predict_sequence_length=3, custom_model_params=None):
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
        """A RNN seq2seq structure for time series

        :param inputs: _description_
        :type inputs: _type_
        :param teacher: teacher forcing decoding, defaults to None
        :type teacher: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, decoder_features = inputs
            encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            decoder_features = None

        encoder_outputs, encoder_state = self.encoder(encoder_features)

        decoder_outputs = self.decoder(
            decoder_features,
            decoder_init_input=x[:, -1, 0:1],
            init_state=encoder_state,
            teacher=teacher,
            scheduler_sampling=self.params["scheduler_sampling"],
            encoder_output=encoder_outputs,
        )

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, :]
            decoder_outputs = decoder_outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x, axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            decoder_outputs = decoder_outputs + x_mean
        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_type, rnn_size, rnn_dropout=0, dense_size=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        if rnn_type.lower() == "gru":
            self.rnn = GRU(
                units=rnn_size, activation="tanh", return_state=True, return_sequences=True, dropout=rnn_dropout
            )
        elif self.rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                units=self.rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=self.rnn_dropout,
            )
        self.dense = Dense(units=dense_size, activation="tanh")

    def call(self, inputs):
        """_summary_

        Parameters
        ----------
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # outputs: batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        outputs, state = self.rnn(inputs)
        state = self.dense(state)
        # encoder_hidden_state = tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        # outputs = self.dense(outputs)  # => batch_size * input_seq_length * dense_size
        return outputs, state


class Decoder1(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_type,
        rnn_size,
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
            self.rnn = LSTMCell(units=self.rnn_size)
        self.dense = Dense(units=1, activation=None)
        if self.use_attention:
            self.attention = FullAttention(
                hidden_size=self.attention_sizes,
                num_heads=self.attention_heads,
                attention_dropout=self.attention_dropout,
            )

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
        """_summary_

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
                att = self.attention(
                    tf.expand_dims(prev_state, 1), k=kwargs["encoder_output"], v=kwargs["encoder_output"]
                )
                att = tf.squeeze(att, 1)
                this_input = tf.concat([this_input, att], axis=-1)

            this_output, this_state = self.rnn_cell(this_input, prev_state)
            prev_state = this_state
            prev_output = self.dense(this_output)
            decoder_outputs.append(prev_output)

        decoder_outputs = tf.concat(decoder_outputs, axis=-1)
        return tf.expand_dims(decoder_outputs, -1)


class Decoder2(tf.keras.layers.Layer):
    def __init__(self, params):
        super(Decoder2, self).__init__()
        self.params = params
        self.rnn_cell = GRUCell(self.params["rnn_size"])
        self.dense = Dense(units=1)
        self.attention = FullAttention(hidden_size=32, num_heads=2, attention_dropout=0.0)

    def forward(
        self,
        decoder_feature,
        init_state,
        decoder_init_value,
        encoder_output,
        predict_seq_length,
        teacher,
        use_attention,
    ):
        def cond_fn(time, prev_output, prev_state, decoder_output_ta):
            return time < predict_seq_length

        def body(time, prev_output, prev_state, decoder_output_ta):
            if time == 0 or teacher is None:
                this_input = prev_output

            else:
                this_input = teacher[:, time - 1, :]

            if decoder_feature is not None:
                this_feature = decoder_feature[:, time, :]
                this_input = tf.concat([this_input, this_feature], axis=1)

            if use_attention:
                attention_feature = self.attention(tf.expand_dims(prev_state[-1], 1), encoder_output, encoder_output)
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
            tf.TensorArray(dtype=tf.float32, size=predict_seq_length),
        ]
        _, _, _, decoder_outputs_ta = tf.while_loop(cond_fn, body, loop_init)

        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs

    def call(
        self,
        decoder_feature,
        init_state,
        decoder_init_input,
        encoder_output,
        predict_seq_length=1,
        teacher=None,
        use_attention=False,
    ):
        """_summary_

        Parameters
        ----------
        decoder_feature : _type_
            _description_
        init_state : _type_
            _description_
        decoder_init_input : _type_
            _description_
        encoder_output : _type_
            _description_
        predict_seq_length : int, optional
            _description_, by default 1
        teacher : _type_, optional
            _description_, by default None
        use_attention : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        return self.forward(
            decoder_feature=decoder_feature,
            init_state=[init_state],  # for tf2
            decoder_init_value=decoder_init_input,
            encoder_output=encoder_output,
            predict_seq_length=predict_seq_length,
            teacher=teacher,
            use_attention=use_attention,
        )


class Decoder3(tf.keras.layers.Layer):
    # multi-steps static decoding
    def __init__(self, rnn_type, rnn_size, rnn_dropout=0, dense_size=1, **kwargs) -> None:
        super(Decoder3, self).__init__()
        if rnn_type.lower() == "gru":
            self.rnn = GRU(
                units=rnn_size, activation="tanh", return_state=False, return_sequences=True, dropout=rnn_dropout
            )
        elif self.rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                units=self.rnn_size,
                activation="tanh",
                return_state=False,
                return_sequences=True,
                dropout=self.rnn_dropout,
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
        """_summary_

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
