from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense


class TimeMixing(tf.keras.layers.Layer):
    """TensorFlow RWKV time mixing"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = config.hidden_size

    def build(self, input_shape):
        self.time_mix_k = self.add_weight(name="time_mix_k", shape=(1, 1, self.n_embd), initializer="zeros")
        self.time_mix_v = self.add_weight(name="time_mix_v", shape=(1, 1, self.n_embd), initializer="zeros")
        self.time_mix_r = self.add_weight(name="time_mix_r", shape=(1, 1, self.n_embd), initializer="zeros")
        self.time_first = self.add_weight(name="time_first", shape=(1, self.n_embd), initializer="zeros")
        self.time_decay = self.add_weight(name="time_decay", shape=(1, self.n_embd), initializer="zeros")

        self.key = Dense(self.n_embd, use_bias=False)
        self.value = Dense(self.n_embd, use_bias=False)
        self.receptance = Dense(self.n_embd, use_bias=False)
        self.output_layer = Dense(self.n_embd, use_bias=False)

    def call(self, x, state):
        """time mixing

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        """
        # state = [last_x, aa, bb, pp]
        last_x, aa, bb, pp = state

        # Shifted x for mixing
        last_x_expanded = tf.expand_dims(last_x, 1)
        # x shape: (Batch, Seq, Hidden)
        if tf.shape(x)[1] > 1:
            xx = tf.concat([last_x_expanded, x[:, :-1, :]], axis=1)
        else:
            xx = last_x_expanded

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        r = tf.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)

        # WKV calculation (recursive)
        # For simplicity/correctness in RNN form, we process along the time dimension
        seq_len = tf.shape(x)[1]

        outputs = tf.TensorArray(tf.float32, size=seq_len)

        curr_aa, curr_bb, curr_pp = aa, bb, pp

        for t in range(seq_len):
            kt = k[:, t, :]
            vt = v[:, t, :]

            # WKV calculation
            ww = self.time_first + kt
            qq = tf.maximum(curr_pp, ww)
            e1 = tf.exp(curr_pp - qq)
            e2 = tf.exp(ww - qq)
            wkv = (e1 * curr_aa + e2 * vt) / (e1 * curr_bb + e2)
            outputs = outputs.write(t, wkv)

            # Update state
            ww = curr_pp + self.time_decay
            qq = tf.maximum(ww, kt)
            e1 = tf.exp(ww - qq)
            e2 = tf.exp(kt - qq)
            curr_aa = e1 * curr_aa + e2 * vt
            curr_bb = e1 * curr_bb + e2
            curr_pp = qq

        wkv_all = tf.transpose(outputs.stack(), [1, 0, 2])  # [B, T, C]

        new_state = [x[:, -1, :], curr_aa, curr_bb, curr_pp]
        return self.output_layer(r * wkv_all), new_state


class ChannelMixing(tf.keras.layers.Layer):
    """TensorFlow RWKV channel mixing"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = config.hidden_size

    def build(self, input_shape):
        self.time_mix_k = self.add_weight(name="time_mix_k", shape=(1, 1, self.n_embd), initializer="zeros")
        self.time_mix_r = self.add_weight(name="time_mix_r", shape=(1, 1, self.n_embd), initializer="zeros")

        self.key = Dense(self.n_embd, use_bias=False)
        self.value = Dense(self.n_embd, use_bias=False)
        self.receptance = Dense(self.n_embd, use_bias=False)

    def call(self, x, state):
        """channel mixing
        # state is the x from the LAST timestep of the PREVIOUS batch: shape (batch, hidden_size)
        # We need to shift x by 1 and prepending the state

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        """

        # state is the x from the LAST timestep of the PREVIOUS batch
        last_x = state

        last_x_expanded = tf.expand_dims(last_x, 1)

        if tf.shape(x)[1] > 1:
            xx = tf.concat([last_x_expanded, x[:, :-1, :]], axis=1)
        else:
            xx = last_x_expanded

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        r = tf.sigmoid(self.receptance(xr))
        k = tf.square(tf.nn.relu(self.key(xk)))
        kv = self.value(k)

        return r * kv, x[:, -1, :]
