
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01


import numpy as np
import tensorflow as tf


class DenseEinsum(tf.keras.layers.Layer):
    def __init__(self,
                 output_shape,
                 num_summed_dimensions=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 ):
        self._output_shape=output_shape
        self._num_summed_dimentions=num_summed_dimensions
        self._activation = tf.keras.activations.get(activation)
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._einsum_string = None
        self._CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
        super(DenseEinsum,self).__init__()

    def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
        input_str = ""
        kernel_str = ""
        output_str = ""
        letter_offset = 0
        for i in range(free_input_dims):
            char = self._CHR_IDX[i + letter_offset]
            input_str += char
            output_str += char

        letter_offset += free_input_dims
        for i in range(bound_dims):
            char = self._CHR_IDX[i + letter_offset]
            input_str += char
            kernel_str += char

        letter_offset += bound_dims
        for i in range(output_dims):
            char = self._CHR_IDX[i + letter_offset]
            kernel_str += char
            output_str += char

        return input_str + "," + kernel_str + "->" + output_str

    def build(self,input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_rank = input_shape.rank
        self._einsum_string=self._build_einsum_string(input_rank-self._num_summed_dimentions,
                                                      self._num_summed_dimentions,
                                                      len(self._output_shape)) #"BTF,FCD->BTCD"

        self._kernel_shape = (input_shape[input_rank-self._num_summed_dimentions:].concatenate(self._output_shape))
        self._kernel = self.add_weight(
                                    "kernel",
                                    shape=self._kernel_shape,
                                    initializer=self._kernel_initializer,
                                    regularizer=self._kernel_regularizer,
                                    constraint=self._kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
        if self._use_bias:
            self._bias = self.add_weight(
                "bias",
                shape=self._output_shape,
                initializer=self._bias_initializer,
                regularizer=self._bias_regularizer,
                constraint=self._bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self._bias = None

        super(DenseEinsum, self).build(input_shape)

    def call(self,inputs):
        ret=tf.einsum(self._einsum_string,inputs,self._kernel)
        if self._use_bias:
            ret += self._bias
        if self._activation is not None:
            ret = self._activation(ret)
        return ret


class Attention(tf.keras.layers.Layer):
    def __init__(self,hidden_size,num_heads,attention_dropout):
        if hidden_size%num_heads:
            raise ValueError("Hidden size ({}) must be divisible by the number of heads ({})."
                             .format(hidden_size, num_heads))

        super(Attention,self).__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.attention_dropout=attention_dropout

    def build(self,input_shape):
        self.query_dense_layer = DenseEinsum([self.num_heads,self.hidden_size//self.num_heads])
        self.key_dense_layer = DenseEinsum([self.num_heads,self.hidden_size//self.num_heads])
        self.value_dense_layer = DenseEinsum([self.num_heads,self.hidden_size//self.num_heads])
        self.output_dense_layer = DenseEinsum([self.hidden_size,],num_summed_dimensions=2)
        super(Attention, self).build(input_shape)

    def forward(self,query_input,source_input,attention_mask,training,cache):
        query=self.query_dense_layer(query_input)
        key=self.key_dense_layer(source_input)
        value=self.value_dense_layer(source_input)

        if cache is not None:
            key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
            value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

            # Update cache
            cache["k"] = key
            cache["v"] = value

        depth = (self.hidden_size // self.num_heads)
        query *= depth ** -0.5
        logits = tf.einsum("BTNH,BFNH->BNFT", key, query)

        if attention_mask is not None:
            logits+=attention_mask

        weights = tf.nn.softmax(logits, name="attention_weights")
        if training:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

        attention_output = self.output_dense_layer(attention_output)
        #print('*'*10,attention_output.shape)
        return attention_output

    def call(self, query_input, source_input,bias,training,cache=None):
        return self.forward(query_input,source_input,bias,training,cache)


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self,hidden_size,filter_size,relu_dropout):
        super(FeedForwardNetwork,self).__init__()
        self.hidden_size=hidden_size
        self.filter_size=filter_size
        self.relu_dropout=relu_dropout
        self.filter_dense_layer = tf.keras.layers.Dense(
                                                        self.filter_size,
                                                        use_bias=True,
                                                        activation=tf.nn.relu,
                                                        name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=True, name="output_layer")

    def forward(self,x,training):
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

    def call(self,x,training):
        return self.forward(x,training)


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self,embedding_size):
        super(EmbeddingLayer,self).__init__()
        self.embedding_size=embedding_size

    def build(self,input_shape):
        with tf.name_scope('embedding'):
            self.shared_weights=self.add_weight(name='weights',
                                                shape=[input_shape[-1],self.embedding_size],
                                                initializer=tf.random_normal_initializer(mean=0.,
                                                                                         stddev=self.embedding_size ** -0.5)
                                                )
        super(EmbeddingLayer,self).build(input_shape)

    def get_config(self):
        return {
            #'vocab_size':self.vocab_size,
            'embedding_size':self.embedding_size
        }

    def call(self,x):
        y=tf.einsum('bsf,fk->bsk',x,self.shared_weights)
        return y


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self,max_len):
        super(PositionEncoding, self).__init__()
        self.max_len=max_len

    def build(self,input_shape):
        super(PositionEncoding,self).build(input_shape)

    def get_config(self):
        return {
            'max_len': self.max_len
        }

    def call(self,x,masking=True):
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
        return tf.cast(outputs,tf.float32)


class SublayerConnection(tf.keras.layers.Layer):
    def __init__(self,sublayer,params):
        super(SublayerConnection,self).__init__()
        self.sublayer=sublayer
        self.params=params
        self.layer_postprocess_dropout=params['layer_postprocess_dropout']

    def build(self,input_shape):
        self.layer_norm=tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        super(SublayerConnection,self).build(input_shape)

    def get_config(self):
        return {
            'params':self.params
        }

    def call(self,x,*args,**kwargs):
        y=self.sublayer(self.layer_norm(x),*args,**kwargs)
        if kwargs['training']:
            y=tf.nn.dropout(y,rate=self.layer_postprocess_dropout)
        return x+y
