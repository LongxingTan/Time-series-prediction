"""
`DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
<https://arxiv.org/abs/1704.04110>`_ (Salinas, Flunkert, Gasthaus & Januschowski, IJF 2020)

TensorFlow implementation matched to the ``pytorch_forecasting`` DeepAR reference used by the
AR parity task (Phase 1):

- per-``series`` static embedding broadcast across every time step,
- multi-layer LSTM (``hidden_size=30``, ``rnn_layers=2``) that at each step consumes the
  lagged (previous) target value concatenated with the static series embedding,
- a distribution head projecting each decoder step to the parameters of a univariate Normal
  (``loc`` plus a softplus-constrained positive ``scale``) -- this makes the model
  probabilistic rather than a plain point-forecast RNN,
- training loss: negative log-likelihood of the true next value under the predicted Normal,
  aggregated over the decoder horizon,
- inference: ancestral sampling (each decoder step samples from the predicted Normal and feeds
  that value back as the next lagged target), with ``n_samples`` paths aggregated to a point
  forecast.

The model is built in **normalized space**: the Phase 2 pipeline feeds (and the model predicts)
``loc``/``scale`` for the standardized target. Un-normalisation happens in the Phase 4 eval /
sampling loop, mirroring ``pytorch_forecasting``'s ``transform_output``.

Input convention (``(x, decoder_feature, static)`` measured in the ``tfts`` 3-tuple style):
- ``x``: ``(batch, encoder_length, 1)`` normalized encoder values
- ``decoder_feature``: ``(batch, prediction_length, 1)`` teacher-forced lagged target, i.e.
  ``[last_encoder_value, y[0], y[1], ..., y[-2]]`` (previous step's value)
- ``static``: ``(batch, 1)`` int series id (fed through an embedding)
"""

from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Embedding, LSTM, Lambda

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


class DeepAR(BaseModel):
    """DeepAR (autoregressive LSTM + univariate Normal head)."""

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

    def _extract(self, inputs):
        """Unpack ``(x, decoder_feature, static)`` from dict/list/tensor."""
        if isinstance(inputs, dict):
            x = inputs.get("x")
            decoder_feature = inputs.get("decoder_feature")
            static = inputs.get("static")
            return x, decoder_feature, static
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

    def __call__(
        self,
        inputs: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        cfg = self.config
        x, decoder_feature, static = self._extract(inputs)

        encoder_length = int(x.shape[1])
        prediction_length = int(decoder_feature.shape[1])
        self.train_sequence_length = encoder_length
        self.config.train_sequence_length = encoder_length
        self.config.predict_sequence_length = prediction_length
        self.predict_sequence_length = prediction_length

        # ---- static series embedding, broadcast across every time step ----
        emb = Embedding(cfg.n_series, cfg.embedding_size, name="series_embedding")(static)  # (B,1,emb)

        # encoder consumes the first encoder_length-1 values (matches PF: RNN over rolled target
        # with the first step dropped; the last encoder value becomes the decoder's seed target).
        enc_seq = self._slice_steps(x, 0, encoder_length - 1)  # (B, enc_len-1, 1)
        dec_seq = decoder_feature  # (B, prediction_length, 1)

        enc_in = Concatenate(axis=-1)([enc_seq, self._tile(emb, encoder_length - 1)])
        dec_in = Concatenate(axis=-1)([dec_seq, self._tile(emb, prediction_length)])
        # (B, (enc_len-1)+prediction_length, 1+emb)
        seq = Concatenate(axis=1)([enc_in, dec_in])

        # ---- multi-layer LSTM (hidden state flows continuously across encoder+decoder) ----
        h = seq
        for i in range(cfg.rnn_layers):
            h = LSTM(
                cfg.hidden_size,
                return_sequences=True,
                dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
                recurrent_dropout=0.0,
                name=f"lstm_{i}",
            )(h)
        # decoder positions (last prediction_length steps)
        dec_start = encoder_length - 1  # total encoder+decoder length minus prediction_length
        h = self._slice_steps(h, dec_start, prediction_length)  # (B, pred, hidden)

        # ---- univariate Normal head: loc + positive scale (softplus) ----
        loc = Dense(1, name="loc")(h)  # (B, pred, 1)
        scale_param = Dense(1, name="scale_param")(h)
        scale = Lambda(lambda t: tf.math.softplus(t) + 1e-6, name="scale")(scale_param)

        return {"loc": loc, "scale": scale} if return_dict else {"loc": loc, "scale": scale}