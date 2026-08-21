"""
`DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
<https://arxiv.org/abs/1704.04110>`_ (Salinas, Flunkert, Gasthaus & Januschowski, IJF 2020)

This module implements DeepAR as a set of *persistent* modules (created once in
``__init__``) so the model can be serialized and supports step-wise autoregressive
generation via ``decode_step`` while keeping a fast, vectorized teacher-forced
training path in ``__call__``.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import RNN, Concatenate, Embedding, Lambda, LSTMCell

from ..distributions import NormalOutput
from ..generation import AutoregressiveGenerationMixin
from .base import BaseConfig, BaseModel


class DeepARConfig(BaseConfig):
    model_type: str = "deep_ar"

    def __init__(
        self,
        hidden_size: int = 30,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        embedding_size: int = 50,
        n_series: int = 100,
    ):
        super(DeepARConfig, self).__init__()
        # defaults match the Phase 1 PyTorch Forecasting DeepAR reference
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.n_series = n_series


class DeepAREncoder(tf.keras.layers.Layer):
    """Runs the (possibly multi-layer) LSTM stack over the encoder window.

    Encapsulates the weight-shared LSTM cells used by both the vectorized training
    path and the step-wise ``decode_step`` generator.
    """

    def __init__(self, config: DeepARConfig, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        # persistent cells: variable names ``lstm_<i>/kernel`` etc. keep the Phase-4
        # checkpoint weight-prefixes compatible.
        dropout = config.dropout if config.rnn_layers > 1 else 0.0
        self.lstm_cells = [
            LSTMCell(config.hidden_size, dropout=dropout, name=f"lstm_{i}") for i in range(config.rnn_layers)
        ]
        # RNN wrappers share the same cells (no re-build); they give the fast full-
        # sequence step and the last-step state, and are what ``__call__`` traces.
        self.lstm_layers = [
            RNN(cell, return_sequences=True, return_state=True, name=f"lstm_rnn_{i}")
            for i, cell in enumerate(self.lstm_cells)
        ]

    def encode(self, seq: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, list]:
        """

        Parameters
        ----------
        seq : (batch, seq_len, feature)  merged encoder+decoder inputs
        training : apply dropout (train path only)

        Returns
        -------
        (last_output, final_states)
        """
        h = seq
        states = []
        for layer in self.lstm_layers:
            out, hh, cc = layer(h, training=training)
            states.append((hh, cc))
            h = out
        return h, states


class DeepAR(BaseModel, AutoregressiveGenerationMixin):
    """DeepAR -- autoregressive LSTM + univariate Normal head.

    Public API
    ----------
    - ``output = model(training_inputs)``  -> ``{"loc", "scale"}`` (teacher-forced, unchanged)
    - ``forecast = model.generate({"x": x, "static": static}, generation_config=...)``
      -> ``ForecastGenerationOutput`` (optional, sampled)
    """

    def __init__(
        self,
        predict_sequence_length: int = 20,
        config: Optional[DeepARConfig] = None,
    ):
        super().__init__(
            predict_sequence_length=predict_sequence_length,
            config=config or DeepARConfig(),
        )
        self.config: DeepARConfig = self.config
        self.train_sequence_length = None

        # ---- persistent modules (built once) ----
        self.series_embedding = Embedding(self.config.n_series, self.config.embedding_size, name="series_embedding")
        self.encoder = DeepAREncoder(self.config, name="deepar_encoder")
        # probabilistic head: univariate Normal (loc + softplus scale), NLL loss.
        self.output_distribution = NormalOutput(target_dim=1)

    # ------------------------------------------------------------------ input
    def _extract(self, inputs):
        """Unpack ``(x, decoder_feature, static)`` from dict/list/tensor."""
        if isinstance(inputs, dict):
            return inputs.get("x"), inputs.get("decoder_feature"), inputs.get("static")
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
            decoder_feature = inputs[1] if len(inputs) > 1 else None
            static = inputs[2] if len(inputs) > 2 else None
            return x, decoder_feature, static
        return inputs, None, None

    @staticmethod
    def _tile(emb, length):
        """Broadcast a ``(batch, 1, emb)`` embedding across ``length`` time steps."""
        return Lambda(
            lambda e: tf.tile(e, [1, length, 1]),
            output_shape=lambda s: (s[0], length, s[-1]),
        )(emb)

    @staticmethod
    def _slice_steps(x, start, length):
        return Lambda(
            lambda t: t[:, start : start + length, :],
            output_shape=lambda s: (s[0], length, s[-1]),
        )(x)

    # -------------------------------------------------- teacher-forced forward
    def __call__(
        self,
        inputs: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ):
        x, decoder_feature, static = self._extract(inputs)

        if x is None or decoder_feature is None or static is None:
            raise ValueError("DeepAR requires 'x', 'decoder_feature', and 'static' inputs.")

        encoder_length = int(x.shape[1])
        prediction_length = int(decoder_feature.shape[1])
        self.train_sequence_length = encoder_length
        self.config.train_sequence_length = encoder_length
        self.config.predict_sequence_length = prediction_length
        self.predict_sequence_length = prediction_length

        emb = self.series_embedding(static)  # (B, 1, emb)
        # encoder consumes the first encoder_length-1 values (matches PF); the last
        # encoder value becomes the decoder's seed target.
        enc_seq = self._slice_steps(x, 0, encoder_length - 1)
        enc_in = Concatenate(axis=-1)([enc_seq, self._tile(emb, encoder_length - 1)])
        dec_in = Concatenate(axis=-1)([decoder_feature, self._tile(emb, prediction_length)])
        seq = Concatenate(axis=1)([enc_in, dec_in])  # (B, enc-1+pred, 1+emb)

        # multi-layer LSTM; hidden state flows continuously across encoder+decoder.
        h, _states = self.encoder.encode(seq, training=training or False)

        dec_start = encoder_length - 1
        h = self._slice_steps(h, dec_start, prediction_length)  # (B, pred, hidden)

        params = self.output_distribution.parameters(h)
        return params if return_dict else params

    # ------------------------------------------- generation hooks (eager path)
    def initialize_generation_state(self, x: tf.Tensor, static: tf.Tensor) -> list:
        """Encode the window and return the per-layer LSTM final state."""
        if x.shape[1] is None:
            raise ValueError("DeepAR generation requires a statically known encoder length.")
        enc_len = int(x.shape[1])
        emb = self.series_embedding(static)  # (B, 1, emb)
        enc_seq = x[:, : enc_len - 1, :]
        enc_in = tf.concat([enc_seq, tf.tile(emb, [1, enc_len - 1, 1])], axis=-1)
        h = enc_in
        states = []
        for layer in self.encoder.lstm_layers:
            out, hh, cc = layer(h, training=False)
            states.append((hh, cc))
            h = out
        return states

    def decode_step(
        self,
        previous_target: tf.Tensor,
        static: tf.Tensor,
        state: list,
        training: bool = False,
    ):
        """One autoregressive decoder step: ``(params, next_state) = decode_step(...)``."""
        emb = self.series_embedding(static)  # (B, 1, emb)
        x = tf.concat([previous_target, emb], axis=-1)  # (B, 1, 1+emb)
        new_states = []
        h = x
        for i, cell in enumerate(self.encoder.lstm_cells):
            inp = tf.squeeze(h, axis=1)  # (B, feat)
            out, [nh, nc] = cell(inp, state[i], training=training)
            new_states.append((nh, nc))
            h = tf.expand_dims(out, axis=1)  # (B, 1, hidden)
        params = self.output_distribution.parameters(h)
        return params, new_states
