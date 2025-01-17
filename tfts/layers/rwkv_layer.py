import tensorflow as tf


class TimeMixing(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = config.n_embd

        # Trainable parameters
        self.time_mix_k = self.add_weight("time_mix_k", (1, self.n_embd), initializer="zeros")
        self.time_mix_v = self.add_weight("time_mix_v", (1, self.n_embd), initializer="zeros")
        self.time_mix_r = self.add_weight("time_mix_r", (1, self.n_embd), initializer="zeros")
        self.time_first = self.add_weight("time_first", (1, self.n_embd), initializer="zeros")
        self.time_decay = self.add_weight("time_decay", (1, self.n_embd), initializer="zeros")

        self.key = tf.keras.layers.Dense(self.n_embd, use_bias=False)
        self.value = tf.keras.layers.Dense(self.n_embd, use_bias=False)
        self.receptance = tf.keras.layers.Dense(self.n_embd, use_bias=False)
        self.output = tf.keras.layers.Dense(self.n_embd, use_bias=False)

    def call(self, x, state):
        aa, bb, pp = state

        # Mix with previous timestep
        xk = x * self.time_mix_k + state[0] * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state[0] * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state[0] * (1 - self.time_mix_r)

        r = tf.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)

        ww = self.time_first + k
        qq = tf.maximum(pp, ww)
        e1 = tf.exp(pp - qq)
        e2 = tf.exp(ww - qq)

        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b

        # Update states
        ww = pp + self.time_decay
        qq = tf.maximum(ww, k)
        e1 = tf.exp(ww - qq)
        e2 = tf.exp(k - qq)

        new_aa = e1 * aa + e2 * v
        new_bb = e1 * bb + e2
        new_pp = qq

        new_state = [new_aa, new_bb, new_pp]

        return self.output(r * wkv), new_state


class ChannelMixing(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = config.n_embd

        self.time_mix_k = self.add_weight("time_mix_k", (1, self.n_embd), initializer="zeros")
        self.time_mix_r = self.add_weight("time_mix_r", (1, self.n_embd), initializer="zeros")

        self.key = tf.keras.layers.Dense(self.n_embd, use_bias=False)
        self.value = tf.keras.layers.Dense(self.n_embd, use_bias=False)
        self.receptance = tf.keras.layers.Dense(self.n_embd, use_bias=False)

    def call(self, x, state):
        xk = x * self.time_mix_k + state * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state * (1 - self.time_mix_r)

        r = tf.sigmoid(self.receptance(xr))
        k = tf.square(tf.nn.relu(self.key(xk)))
        return r * self.value(k), x
