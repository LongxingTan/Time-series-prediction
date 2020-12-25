# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper:
# other implementations: https://github.com/Arturus/kaggle-web-traffic

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRUCell, LSTMCell, RNN
from deepts.layers.attention_layer import Attention


params = {
    'rnn_size': 64,
    'dense_size': 16,
    'num_stacked_layers': 1,
    'use_attention': False,
}


class Seq2seq(object):
    def __init__(self, custom_model_params):
        params.update(custom_model_params)
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.params = params

    def __call__(self, inputs, training, predict_seq_length, teacher=None):
        if isinstance(inputs, tuple):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:  # for single variable prediction
            encoder_feature = x = inputs
            decoder_feature = None

        encoder_output, encoder_state = self.encoder(encoder_feature)
        decoder_output = self.decoder(decoder_feature, encoder_state, x,
                                      encoder_output=encoder_output,
                                      predict_seq_length=predict_seq_length,
                                      teacher=teacher,
                                      use_attention=self.params['use_attention'])
        return decoder_output


class Encoder(object):
    def __init__(self, params):
        self.params = params
        cell = GRUCell(units=self.params['rnn_size'])
        self.rnn = RNN(cell, return_state=True, return_sequences=True)
        self.dense = Dense(units=1)

    def __call__(self, inputs, training=None, mask=None):
        # outputs: batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        outputs, state = self.rnn(inputs)
        #encoder_hidden_state=tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        outputs = self.dense(outputs)  # => batch_size * input_seq_length * dense_size
        return outputs, state


class Decoder(object):
    def __init__(self, params):
        self.params = params
        self.rnn_cell = GRUCell(self.params['rnn_size'])
        self.rnn = RNN(self.rnn_cell, return_state=True, return_sequences=True)
        self.dense = Dense(units=1)
        self.attention = Attention(hidden_size=32, num_heads=2, attention_dropout=0.8)

    def forward(self, decoder_feature, init_state, decoder_init_value,
                encoder_output, predict_seq_length, teacher, use_attention):

        def cond_fn(time, prev_output, prev_state, decoder_output_ta):
            return time < predict_seq_length

        def body(time, prev_output, prev_state, decoder_output_ta):
            if time == 0 or teacher is None:
                this_input = prev_output
            else:
                this_input = teacher[:, time-1, :]

            if decoder_feature is not None:
                this_feature = decoder_feature[:, time, :]
                this_input = tf.concat([this_input, this_feature], axis=1)

            if use_attention:
                attention_feature = self.attention(tf.expand_dims(prev_state[-1], 1), encoder_output)
                attention_feature = tf.squeeze(attention_feature, 1)
                this_input = tf.concat([this_input, attention_feature], axis=-1)

            this_output, this_state = self.rnn_cell(this_input, prev_state)
            project_output = self.dense(this_output)
            decoder_output_ta = decoder_output_ta.write(time, project_output)
            return time+1, project_output, this_state, decoder_output_ta

        loop_init = [tf.constant(0, dtype=tf.int32),  # steps
                     decoder_init_value,  # decoder each step
                     init_state,  # state
                     tf.TensorArray(dtype=tf.float32, size=predict_seq_length)]
        _, _, _, decoder_outputs_ta = tf.while_loop(cond_fn, body, loop_init)

        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        #decoder_outputs=decoder_outputs
        return decoder_outputs

    def __call__(self, decoder_feature, encoder_state, x, encoder_output,
                 predict_seq_length=1, teacher=None, use_attention=False):

        #context_vector=self.attention(x)
        #x=tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)
        #output,state=self.rnn(x)
        #x=self.dense(output)
        return self.forward(decoder_feature=decoder_feature,
                            init_state=[encoder_state],  # for tf2
                            decoder_init_value=x[:, -1, :],
                            encoder_output=encoder_output,
                            predict_seq_length=predict_seq_length,
                            teacher=teacher,
                            use_attention=use_attention)

