# -*- coding: utf-8 -*-
# @author: Longxing Tan
"""Layer for :py:class:`~tfts.models.transformer` :py:class:`~tfts.models.autoformer`"""

import math
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization

from tfts.layers.mask_layer import ProbMask


class FullAttention(tf.keras.layers.Layer):
    """Multi-head attention layer"""

    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0):
        """Initialize the layer.

        Parameters:
        -----------
        hidden_size : int
            The number of hidden units in each attention head.
        num_heads : int
            The number of attention heads.
        attention_dropout : float, optional
            Dropout rate for the attention weights. Defaults to 0.0.
        """
        super(FullAttention, self).__init__()
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({}).".format(hidden_size, num_heads)
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        self.dense_q = Dense(self.hidden_size, use_bias=False)
        self.dense_k = Dense(self.hidden_size, use_bias=False)
        self.dense_v = Dense(self.hidden_size, use_bias=False)
        self.dropout = Dropout(rate=self.attention_dropout)
        super(FullAttention, self).build(input_shape)

    def call(self, q, k, v, mask=None):
        """use query and key generating an attention multiplier for value, multi_heads to repeat it

        Parameters
        ----------
        q : tf.Tenor
            Query with shape batch * seq_q * fea
        k : tf.Tensor
            Key with shape batch * seq_k * fea
        v : tf.Tensor
            value with shape batch * seq_v * fea
        mask : _type_, optional
            important to avoid the leaks, defaults to None, by default None

        Returns
        -------
        tf.Tensor
            tensor with shape batch * seq_q * (units * num_heads)
        """
        q = self.dense_q(q)  # project the query/key/value to num_heads * units
        k = self.dense_k(k)
        v = self.dense_v(v)

        q_ = tf.concat(tf.split(q, self.num_heads, axis=2), axis=0)  # multi-heads transfer to multi-sample
        k_ = tf.concat(tf.split(k, self.num_heads, axis=2), axis=0)
        v_ = tf.concat(tf.split(v, self.num_heads, axis=2), axis=0)

        score = tf.linalg.matmul(q_, k_, transpose_b=True)  # => (batch * heads) * seq_q * seq_k
        score /= tf.cast(tf.shape(q_)[-1], tf.float32) ** 0.5

        if mask is not None:
            score = score * tf.cast(mask, tf.float32)

        score = tf.nn.softmax(score)
        score = self.dropout(score)

        outputs = tf.linalg.matmul(score, v_)  # (batch * heads) * seq_q * units
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
        return outputs

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }
        base_config = super(FullAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0, **kwargs):
        super(SelfAttention, self).__init__()
        self.attention = FullAttention(hidden_size, num_heads, attention_dropout=attention_dropout)

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        """_summary_

        Parameters
        ----------
        x : tf.Tensor
            input tensor for self-attention
        mask : _type_, optional
            masked, by default None

        Returns
        -------
        tf.Tensor
            _description_
        """
        return self.attention(x, x, x, mask)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        return base_config


class ProbAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int = 128, num_heads: int = 1, attention_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.mask_flag = True
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.factor = 5
        self.scale = None

    def build(self, input_shape):
        self.dense_q = Dense(self.hidden_size, use_bias=False)
        self.dense_k = Dense(self.hidden_size, use_bias=False)
        self.dense_v = Dense(self.hidden_size, use_bias=False)
        super().build(input_shape)

    def _prob_qk(self, q, k, sample_k, top_n):
        B, H, L, E = k.shape
        _, _, S, _ = q.shape

        k_expand = tf.broadcast_to(tf.expand_dims(k, -3), (B, H, L, S, E))
        k_random_index = tf.random.uniform((S, sample_k), maxval=L, dtype=tf.int32)
        k_random_index = tf.tile(k_random_index[tf.newaxis, tf.newaxis, :], [B, H, 1, 1])

        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis, tf.newaxis], (1, H, L, k_random_index.shape[-1]))
        head_indexes = tf.tile(tf.range(H)[tf.newaxis, :, tf.newaxis, tf.newaxis], (B, 1, L, k_random_index.shape[-1]))
        k_indexes = tf.tile(tf.range(L)[tf.newaxis, tf.newaxis, :, tf.newaxis], (B, H, 1, k_random_index.shape[-1]))

        k_random_index = tf.stack([batch_indexes, head_indexes, k_indexes, k_random_index], axis=-1)
        k_sample = tf.gather_nd(k_expand, k_random_index)

        qk_sample = tf.squeeze(tf.matmul(tf.expand_dims(q, -2), tf.transpose(k_sample, [0, 1, 2, 4, 3])))
        m = tf.math.reduce_max(qk_sample, axis=-1) - tf.divide(tf.reduce_sum(qk_sample, axis=-1), L)
        m_top = tf.math.top_k(m, top_n, sorted=False)[1]
        m_top = m_top[tf.newaxis] if B == 1 else m_top
        m_top = tf.tile(m_top, (1, 1, 1))

        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, H, top_n))
        head_indexes = tf.tile(tf.range(H)[tf.newaxis, :, tf.newaxis], (B, 1, top_n))
        idx = tf.stack([batch_indexes, head_indexes, m_top], axis=-1)

        q_reduce = tf.gather_nd(q, idx)
        qk = tf.matmul(q_reduce, tf.transpose(k, (0, 1, 3, 2)))
        return qk, m_top

    def _get_initial_context(self, v, L_Q):
        B, H, L_V, D = v.shape
        if not self.mask_flag:
            v_sum = tf.math.reduce_sum(v, axis=-2)
            context = tf.identity(tf.boradcast_to(tf.expand_dims(v_sum, -2), [B, H, L_Q, v_sum.shape[-1]]))
        else:
            assert L_Q == L_V
            context = tf.math.cumsum(v, axis=-2)
        return context

    def _update_context(self, context_in, v, scores, index, L_Q):
        B, H, L_V, D = v.shape
        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, H, index.shape[-1]))
        head_indexes = tf.tile(tf.range(H)[tf.newaxis, :, tf.newaxis], (B, 1, index.shape[-1]))
        index = tf.stack([batch_indexes, head_indexes, index], axis=-1)

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores).mask
            scores = tf.where(attn_mask, -np.inf, scores)

        attn = tf.nn.softmax(scores, axis=-1)
        context_in = tf.tensor_scatter_nd_update(context_in, index, tf.matmul(attn, v))
        return tf.convert_to_tensor(context_in)

    def call(self, q, k, v, mask=None):
        """Prob attention"""
        q = self.dense_q(q)  # project the query/key/value to num_heads * units
        k = self.dense_k(k)
        v = self.dense_v(v)

        B, L, D = q.shape
        _, S, _ = k.shape

        q_ = tf.reshape(q, (B, self.num_heads, L, -1))
        k_ = tf.reshape(k, (B, self.num_heads, S, -1))
        v_ = tf.reshape(v, (B, self.num_heads, S, -1))

        u_q = self.factor * np.ceil(np.log(L)).astype("int").item()
        u_k = self.factor * np.ceil(np.log(S)).astype("int").item()
        u_q = u_q if u_q < L else L
        u_k = u_k if u_k < S else S

        scores_top, index = self._prob_qk(q_, k_, u_k, u_q)
        scores_top = scores_top * 1.0 / np.sqrt(D // self.num_heads)

        context = self._get_initial_context(v_, L)
        context = self._update_context(context, v_, scores_top, index, L)

        out = tf.reshape(context, (B, L, -1))
        return out

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseAttention(tf.keras.layers.Layer):
    """
    SparseAttention implementation
    """

    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0, **kwargs):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, mask=None):
        """Sparse attention

        Parameters
        ----------
        x : tf.Tensor
            _description_
        mask : tf.Tensor, optional
            _description_, by default None
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config


class FastAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, mask=None):
        """Fast attention

        Parameters
        ----------
        x : tf.Tensor
            _description_
        mask : _type_, optional
            _description_, by default None
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config
