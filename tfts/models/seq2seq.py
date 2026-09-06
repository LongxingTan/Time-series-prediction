"""
`Sequence to Sequence Learning with Neural Networks
<https://arxiv.org/abs/1409.3215>`_
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, GRUCell, LSTMCell

from tfts.generation import PointSampler, StepOutput, TimeAxisEngine
from tfts.generation.decoding import DecodeSession
from tfts.generation.feedback import TeacherForcingPolicy
from tfts.layers.attention_layer import Attention

from ._autoregressive import AUTOREGRESSIVE_CAPABILITIES, AutoregressiveModel, decoder_features, encoder_features
from .base import CommonConfig
from .registry import register_model

logger = logging.getLogger(__name__)


class Seq2seqConfig(CommonConfig):
    model_type: str = "seq2seq"

    def __init__(
        self,
        rnn_hidden_size=64,
        rnn_type="gru",
        bi_direction=False,
        dense_hidden_size=64,
        num_stacked_layers=1,
        scheduled_sampling=0,
        use_attention=False,
        attention_size=64,
        num_attention_heads=2,
        attention_probs_dropout_prob=0,
        target_dim=1,
    ):
        super(Seq2seqConfig, self).__init__()
        self.target_dim = target_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_type = rnn_type
        self.bi_direction = bi_direction
        self.dense_hidden_size = dense_hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.scheduled_sampling = scheduled_sampling  # 0: teacher forcing
        self.use_attention = use_attention
        self.attention_size = attention_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        if self.use_attention:
            assert self.attention_size == self.dense_hidden_size


@register_model(
    "seq2seq", config=Seq2seqConfig, tags=("baseline", "encoder-decoder"), capabilities=AUTOREGRESSIVE_CAPABILITIES
)
class Seq2seq(AutoregressiveModel):
    """Seq2seq model for time series prediction with configurable encoder-decoder architectures."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[Seq2seqConfig] = None):
        super(Seq2seq, self).__init__()
        self.config = config or Seq2seqConfig()
        self.predict_sequence_length = predict_sequence_length

        self.encoder = Encoder(
            rnn_size=self.config.rnn_hidden_size,
            rnn_type=self.config.rnn_type,
            dense_size=self.config.dense_hidden_size,
        )
        self.decoder = Decoder(
            rnn_size=self.config.dense_hidden_size,
            target_dim=self.config.target_dim,
            rnn_type=self.config.rnn_type,
            predict_sequence_length=predict_sequence_length,
            use_attention=self.config.use_attention,
            attention_size=self.config.attention_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )

    def initialize_decode(self, batch, *, horizon, training=False):
        features = decoder_features(batch, horizon)
        memory, state = self.encoder(encoder_features(batch), training=training)
        state = (state,) if self.config.rnn_type.lower() == "gru" else tuple(state)
        if not self.decoder.built:
            self.decoder.build(features.shape)
        return DecodeSession((features, memory), state, self.decoder_seed(batch))

    def decode_step(self, previous, state, context, *, offset, training=False):
        features, memory = context
        return self.decoder.step(previous, state, features[:, offset, :], memory, training=training)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_size, rnn_type="gru", rnn_dropout=0, dense_size=32, return_state=False, **kwargs):
        super().__init__(**kwargs)
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type.lower()
        self.rnn_dropout = rnn_dropout
        self.dense_size = dense_size
        self.return_state = return_state

    def build(self, input_shape):
        super(Encoder, self).build(input_shape)
        if self.rnn_type == "gru":
            self.rnn = GRU(
                units=self.rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=self.rnn_dropout,
                reset_after=False,
            )
        elif self.rnn_type == "lstm":
            self.rnn = LSTM(
                units=self.rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=self.rnn_dropout,
            )
        else:
            raise ValueError(f"No supported RNN type: {self.rnn_type}")

        self.dense = Dense(units=self.dense_size, activation="tanh")
        self.rnn.build(input_shape)
        self.dense.build([input_shape[0], self.rnn_size])
        self.built = True

    def call(self, inputs, training=None):
        """Process input through the encoder RNN and dense layers.

        :param inputs: 3D Input tensor with shape (batch_size, seq_len, num_features)
        :return: Encoder outputs and state.

        outputs : tf.Tensor
            (batch_size, input_sequence_length, rnn_size)
        state : tf.Tensor or tuple of tf.Tensor
            Processed state(s) from the RNN:
            - For GRU: (batch_size, dense_size)
            - For LSTM: tuple of (batch_size, dense_size), (batch_size, dense_size)
        """
        if self.rnn_type == "gru":
            rnn_outputs = self.rnn(inputs, training=training)
            outputs, state = rnn_outputs
            state = self.dense(state)
        elif self.rnn_type == "lstm":
            outputs, state_h, state_c = self.rnn(inputs, training=training)
            state_h = self.dense(state_h)
            state_c = self.dense(state_c)
            state = (state_h, state_c)
        else:
            raise ValueError(f"No supported rnn type of {self.rnn_type}")
        # encoder_hidden_state = tuple(self.dense(hidden_state) for _ in range(config['num_stacked_layers']))
        # outputs = self.dense(outputs)  # => batch_size * input_seq_length * dense_size
        return outputs, state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rnn_size": self.rnn_size,
                "rnn_type": self.rnn_type,
                "rnn_dropout": self.rnn_dropout,
                "dense_size": self.dense_size,
                "return_state": self.return_state,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, _ = input_shape
        rnn_output_shape = (batch_size, seq_len, self.rnn_size)

        # State shape depends on RNN type
        if self.rnn_type == "gru":
            state_shape = (batch_size, self.dense_size)
        elif self.rnn_type == "lstm":
            state_shape = ((batch_size, self.dense_size), (batch_size, self.dense_size))
        else:
            raise ValueError(f"No supported rnn type of {self.rnn_type}")
        return rnn_output_shape, state_shape


class Decoder(tf.keras.layers.Layer):
    """One recurrent transition, used by every decoding policy."""

    def __init__(
        self,
        rnn_size=32,
        rnn_type="gru",
        predict_sequence_length=3,
        use_attention=False,
        attention_size=32,
        num_attention_heads=1,
        attention_probs_dropout_prob=0.0,
        target_dim=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type.lower()
        self.predict_sequence_length = predict_sequence_length
        self.use_attention = use_attention
        self.attention_size = attention_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.target_dim = target_dim
        cells = {"gru": GRUCell, "lstm": LSTMCell}
        if self.rnn_type not in cells:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        self.rnn_cell = cells[self.rnn_type](rnn_size)
        self.dense = Dense(target_dim)
        self.attention = (
            Attention(attention_size, num_attention_heads, attention_probs_dropout_prob) if use_attention else None
        )

    def build(self, input_shape):
        width = input_shape[-1] + self.target_dim + (self.attention_size if self.use_attention else 0)
        self.rnn_cell.build([input_shape[0], width])
        self.dense.build([input_shape[0], self.rnn_size])
        super().build(input_shape)

    def step(self, previous, state, features, memory=None, training=None):
        inputs = [previous[:, 0, :], features]
        if self.attention is not None:
            if memory is None:
                raise ValueError("attention requires encoder memory")
            query = tf.concat(state, axis=-1)[:, None, :]
            inputs.append(self.attention(query, memory, memory, training=training)[:, 0, :])
        hidden, state = self.rnn_cell(tf.concat(inputs, axis=-1), state, training=training)
        return StepOutput(self.dense(hidden)[:, None, :], state=tuple(state))

    def call(
        self,
        decoder_features,
        decoder_init_input,
        init_state,
        teacher=None,
        scheduled_sampling=0,
        training=None,
        **kwargs,
    ):
        state = (init_state,) if tf.is_tensor(init_state) else tuple(init_state)

        def step(previous, state, offset):
            return self.step(
                previous, state, decoder_features[:, offset, :], kwargs.get("encoder_output"), training=training
            )

        probability = 1.0 - scheduled_sampling if teacher is not None else 0.0
        return (
            TimeAxisEngine(PointSampler())
            .run(
                step,
                decoder_init_input[:, None, :],
                state,
                self.predict_sequence_length,
                teacher=teacher,
                teacher_forcing_policy=TeacherForcingPolicy(probability),
            )
            .predictions
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                name: getattr(self, name)
                for name in (
                    "rnn_size",
                    "rnn_type",
                    "predict_sequence_length",
                    "use_attention",
                    "attention_size",
                    "num_attention_heads",
                    "attention_probs_dropout_prob",
                    "target_dim",
                )
            }
        )
        return config


# Import compatibility only: there is one implementation and one decoding loop.
DecoderV1 = Decoder
DecoderV2 = Decoder
