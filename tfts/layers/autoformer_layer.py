# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.autoformer`"""

import math

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling1D, Conv1D, Dense, Dropout


class SeriesDecomp(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = AveragePooling1D(pool_size=kernel_size, strides=1, padding="same")

    def call(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x_ma = self.moving_avg(x)
        return x - x_ma, x_ma

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AutoCorrelation(tf.keras.layers.Layer):
    """Auto"""

    def __init__(self, d_model: int, num_heads: int, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        self.wq = Dense(self.d_model, name="q")
        self.wk = Dense(self.d_model, name="k")
        self.wv = Dense(self.d_model, name="v")
        self.drop = Dropout(self.attention_dropout)
        self.dense = Dense(self.d_model, name="project")

    def time_delay_agg(self, q, k, v):  # TODO: v not used in process
        batch_size = tf.shape(q)[0]
        time_steps = tf.shape(q)[2]
        q_fft = tf.signal.rfft(tf.transpose(q, perm=[0, 1, 3, 2]))
        k_fft = tf.signal.rfft(tf.transpose(k, perm=[0, 1, 3, 2]))
        S_qk = q_fft * tf.math.conj(k_fft)
        R_qk = tf.signal.irfft(S_qk)

        init_index = tf.reshape(tf.range(time_steps), (1, 1, 1, -1))
        init_index = tf.tile(init_index, [batch_size, self.num_heads, self.depth, 1])
        top_k = int(2 * tf.math.log(tf.cast(time_steps, tf.float32)))
        # mean_value = tf.reduce_mean(R_qk, axis=1)
        weights, indices = tf.math.top_k(R_qk, top_k)

        tmp_corr = tf.nn.softmax(weights, axis=-1)
        # tmp_corr = tf.reshape(tmp_corr, (batch_size, 1, 32, -1))

        tmp_values = tf.tile(tf.transpose(q, perm=[0, 1, 3, 2]), tf.constant([1, 1, 1, 2]))
        delays_agg = tf.zeros_like(tf.transpose(q, perm=[0, 1, 3, 2]))

        for i in range(top_k):
            pattern = tf.gather(tmp_values, init_index + tf.expand_dims(indices[..., i], -1), axis=-1, batch_dims=-1)
            # print(pattern.shape, tmp_corr.shape)
            delays_agg = delays_agg + pattern * (tf.expand_dims(tmp_corr[..., i], axis=-1))
        return delays_agg

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, timesteps, depth)

    def call(self, q, k, v, dynamic=True):
        """_summary_

        Parameters
        ----------
        q : _type_
            _description_
        k : _type_
            _description_
        v : _type_
            _description_
        dynamic : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        batch_size = tf.shape(q)[0]

        q = self.drop(self.wq(q))
        k = self.drop(self.wk(k))
        v = self.drop(self.wv(v))

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        L = tf.shape(q)[2]
        S = tf.shape(v)[2]

        if tf.math.greater(L, S):
            zeros = tf.zeros_like(q[:, :, : (L - S), :])
            v = tf.concat([v, zeros], axis=2)
            k = tf.concat([k, zeros], axis=2)
        else:
            v = v[:, :, :L, :]
            k = k[:, :, :L, :]

        delays_agg = self.time_delay_agg(q, k, v)
        delays_agg = tf.transpose(delays_agg, [0, 3, 1, 2])
        concat_delays_agg = tf.reshape(delays_agg, (batch_size, -1, self.d_model))
        output = self.dense(concat_delays_agg)
        return output
