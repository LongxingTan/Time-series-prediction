# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, Dense, LayerNormalization


class Attention(tf.keras.layers.Layer):
    """ Multi-head attention layer

    """
    def __init__(self, hidden_size, num_heads, attention_dropout=0.):
        if hidden_size % num_heads:
            raise ValueError("Hidden size ({}) must be divisible by the number of heads ({})."
                             .format(hidden_size, num_heads))
        super(Attention, self).__init__()
        self.units = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.dense_q = Dense(self.units, use_bias=False)
        self.dense_k = Dense(self.units, use_bias=False)
        self.dense_v = Dense(self.units, use_bias=False)
        self.dropout = Dropout(rate=self.attention_dropout)

    def call(self, q, k, v, mask=None):
        """
        use query and key generating an attention multiplier for value, multi_heads to repeat it
        Args:
            Query: batch * seq_q * fea
            Key: batch * seq_k * fea
            Value: batch * seq_v * fea
        Returns:
            output: batch * key_sequence * (units * num_heads)
        """
        q = self.dense_q(q)  # project the query/key/value to num_heads * units
        k = self.dense_k(k)
        v = self.dense_v(v)

        q_ = tf.concat(tf.split(q, self.num_heads, axis=2), axis=0)  # multi-heads transfer to
        k_ = tf.concat(tf.split(k, self.num_heads, axis=2), axis=0)
        v_ = tf.concat(tf.split(v, self.num_heads, axis=2), axis=0)

        score = tf.linalg.matmul(q_, k_, transpose_b=True)  # => (batch*heads) * seq_q * seq_k
        score /= tf.cast(tf.shape(q_)[-1], tf.float32) ** 0.5

        if mask:
            score = score * tf.cast(mask, tf.float32)

        score = tf.nn.softmax(score)
        score = self.dropout(score)

        outputs = tf.linalg.matmul(score, v_)  # (batch*heads) * seq_q * units
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
        return outputs


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, attention_dropout=0.):
        super(SelfAttention, self).__init__()
        self.attention = Attention(hidden_size, num_heads, attention_dropout=attention_dropout)

    def call(self, x, mask=None):
        return self.attention(x, x, x, mask)


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, filter_size, relu_dropout):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.filter_dense_layer = Dense(self.filter_size, use_bias=True, activation='relu')
        self.output_dense_layer = Dense(self.hidden_size, use_bias=True)

    def forward(self, x, training):
        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)
        return output

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }

    def call(self, x, training):
        return self.forward(x, training)


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.token_weights = self.add_weight(name='token_weights',
                                             shape=[input_shape[-1], self.embedding_size],
                                             initializer=tf.random_normal_initializer(mean=0.,
                                                                                      stddev=self.embedding_size ** -0.5))
        super(TokenEmbedding, self).build(input_shape)

    def get_config(self):
        return {
            'embedding_size': self.embedding_size
        }

    def call(self, x):
        y = tf.einsum('bsf,fk->bsk', x, self.token_weights)
        return y


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)

    def get_config(self):
        return {
            'max_len': self.max_len
        }

    def call(self, x, masking=True):
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic

        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # => batch_size*seq_length
        position_enc = np.array(
            [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen,E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape):
        super(PositionEncoding, self).build(input_shape)

    def get_config(self):
        return {
            'max_len': self.max_len
        }

    def call(self, x, masking=True):
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic
        with tf.name_scope('position_encode'):
            position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # => batch_size*seq_length
            position_enc = np.array(
                [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)])

            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen,E)

            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            if masking:
                outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)


class DataEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(d_model)
        self.position_embedding = PositionEmbedding(d_model)
        self.dropout = Dropout(0.0)

    def call(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
