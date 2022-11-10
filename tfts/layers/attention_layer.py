# -*- coding: utf-8 -*-
# @author: Longxing Tan
"""Layer for :py:class:`~tfts.models.transformer` :py:class:`~tfts.models.autoformer`"""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization


class FullAttention(tf.keras.layers.Layer):
    """Multi-head attention layer"""

    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0):
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({}).".format(hidden_size, num_heads)
            )
        super(FullAttention, self).__init__()
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
        q : _type_
            Query with shape batch * seq_q * fea
        k : _type_
            Key with shape batch * seq_k * fea
        v : _type_
            value with shape batch * seq_v * fea
        mask : _type_, optional
            important to avoid the leaks, defaults to None, by default None

        Returns
        -------
        _type_
            tensor with shape batch * seq_q * (units * num_heads)
        """
        q = self.dense_q(q)  # project the query/key/value to num_heads * units
        k = self.dense_k(k)
        v = self.dense_v(v)

        q_ = tf.concat(tf.split(q, self.num_heads, axis=2), axis=0)  # multi-heads transfer to
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
    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0):
        super(SelfAttention, self).__init__()
        self.attention = FullAttention(hidden_size, num_heads, attention_dropout=attention_dropout)

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        mask : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        return self.attention(x, x, x, mask)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        return base_config


class SparseAttention(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config


class ProbAttention(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, x_mask=None):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        x_mask : _type_, optional
            _description_, by default None
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config


class FastAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, x_mask=None):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        x_mask : _type_, optional
            _description_, by default None
        """
        return

    def get_config(self):
        base_config = super().get_config()
        return base_config
