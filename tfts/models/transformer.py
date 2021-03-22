#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper: https://arxiv.org/pdf/1706.03762.pdf
# other implementations: https://github.com/maxjcohen/transformer
#                        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_bert.py
#                        https://github.com/facebookresearch/detr
#                        https://github.com/zhouhaoyi/Informer2020


import tensorflow as tf
from ..layers.attention_layer import *


params = {
    'n_encoder_layers': 3,
    'n_decoder_layers': 2,
    'attention_hidden_size': 32*4,
    'num_heads': 4,
    'ffn_hidden_size': 32*4,
    'ffn_filter_size': 32*4,
    'attention_dropout': 0.1,
    'relu_dropout': 0.1,
    'layer_postprocess_dropout': 0.1,
}


class Transformer(object):
    def __init__(self, custom_model_params, dynamic_decoding=True):
        params.update(custom_model_params)
        self.params = params
        self.encoder_embedding = DataEmbedding(d_model=params['attention_hidden_size'])
        self.decoder_embedding = DataEmbedding(d_model=params['attention_hidden_size'])
        self.encoder_stack = EncoderStack(self.params)
        self.decoder_stack = DecoderStack(self.params)
        self.projection = tf.keras.layers.Dense(units=1, name='final_proj')

    def get_config(self):
        return {
            'params': self.params
        }

    def __call__(self, inputs, training, predict_seq_length):
        inputs, teacher = inputs

        if isinstance(inputs, tuple):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs
            decoder_feature = None

        memory = self.encoder(encoder_feature, mask=None, training=training)

        if training:
            if teacher is not None:
                # Shift targets to the right, and remove the last element
                # decoder_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
                decoder_inputs = tf.concat([x[:, -1:, :], teacher[:, :-1, :]], axis=1)
            if decoder_feature is not None:
                decoder_inputs = tf.concat([decoder_inputs, decoder_feature], axis=-1)

            decoder_output = self.decoder(decoder_inputs, memory, src_mask=None, training=training, predict_seq_length=predict_seq_length)
            outputs = self.projection(decoder_output)
            return outputs
        else:
            decoder_inputs = decoder_inputs_update = tf.cast(inputs[:, -1:, 0:1], tf.float32)

            for i in range(predict_seq_length):
                if decoder_feature is not None:
                    decoder_inputs_update = tf.concat([decoder_inputs_update, decoder_feature[:, :i+1, :]], axis=-1)
                decoder_inputs_update = self.decoder(decoder_inputs_update, memory, src_mask=None, training=False, predict_seq_length=1)
                decoder_inputs_update = self.projection(decoder_inputs_update)
                decoder_inputs_update = tf.concat([decoder_inputs, decoder_inputs_update], axis=1)
            return decoder_inputs_update[:, 1:, :]

    def encoder(self, encoder_inputs, mask, training):
        """
        :param inputs: sequence_inputs, batch_size * sequence_length * feature_dim
        :param training:
        :return:
        """
        encoder_embedding = self.encoder_embedding(encoder_inputs)  # batch * seq * embedding_size
        return self.encoder_stack(encoder_embedding, mask, training)

    def decoder(self, decoder_inputs, memory, src_mask, training, predict_seq_length):
        decoder_embedding = self.decoder_embedding(decoder_inputs)
        with tf.name_scope("shift_targets"):
            tgt_mask = self.get_tgt_mask_bias(predict_seq_length)
        logits = self.decoder_stack(decoder_embedding, memory, src_mask, tgt_mask, training)  # Todo：mask
        return logits

    def get_src_mask(self, x, pad=0):
        src_mask = tf.reduce_all(tf.math.equal(x, pad), axis=-1)
        return src_mask

    def get_src_mask_bias(self, mask):
        attention_bias = tf.cast(mask, tf.float32)
        attention_bias = attention_bias * tf.constant(-1e9, dtype=tf.float32)
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, 1), 1)  # => batch_size * 1 * 1 * input_length
        return attention_bias

    def get_tgt_mask_bias(self, length):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=tf.float32), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = -1e9 * (1.0 - valid_locs)
        return decoder_bias


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.params['n_encoder_layers']):
            attention_layer = SelfAttention(self.params['attention_hidden_size'],
                                            self.params['num_heads'],
                                            self.params['attention_dropout'])
            feed_forward_layer = FeedForwardNetwork(self.params['ffn_hidden_size'],
                                                    self.params['ffn_filter_size'],
                                                    self.params['relu_dropout'])
            ln_layer1 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=1e-6, dtype="float32")

            self.layers.append([attention_layer, feed_forward_layer, ln_layer1, ln_layer2])
        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, encoder_inputs, training, src_mask=None):
        x = encoder_inputs
        for n, layer in enumerate(self.layers):
            attention_layer, ffn_layer, ln_layer1, ln_layer2 = layer
            x0 = x
            x1 = attention_layer(x0)
            x1 += x0
            x1 = ln_layer1(x1)
            x2 = ffn_layer(x1, training=training)
            x2 += x1
            x = ln_layer2(x2)
        return x


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.params['n_decoder_layers']):
            self_attention_layer = SelfAttention(self.params['attention_hidden_size'],
                                                 self.params['num_heads'],
                                                 self.params['attention_dropout'])
            enc_dec_attention_layer = Attention(self.params['attention_hidden_size'],
                                                self.params['num_heads'],
                                                self.params['attention_dropout'])
            feed_forward_layer = FeedForwardNetwork(self.params['ffn_hidden_size'],
                                                    self.params['ffn_filter_size'],
                                                    self.params['relu_dropout'])
            ln_layer1 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer3 = LayerNormalization(epsilon=1e-6, dtype="float32")

            self.layers.append([self_attention_layer, enc_dec_attention_layer, feed_forward_layer, ln_layer1, ln_layer2, ln_layer3])
        super(DecoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, decoder_inputs, encoder_outputs, src_mask, tgt_mask, training, cache=None):
        x = decoder_inputs

        for n, layer in enumerate(self.layers):
            self_attention_layer, enc_dec_attention_layer, ffn_layer, ln_layer1, ln_layer2, ln_layer3 = layer
            x0 = x
            x1 = self_attention_layer(x0)
            x1 += x0
            x1 = ln_layer1(x1)
            x2 = enc_dec_attention_layer(x1, encoder_outputs, encoder_outputs)
            x2 += x1
            x2 = ln_layer2(x2)
            x3 = ffn_layer(x2, training=training)
            x3 += x2
            x = ln_layer3(x3)
        return x
