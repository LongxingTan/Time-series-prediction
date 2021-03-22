# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper:
# other implementations: https://github.com/sjvasquez/web-traffic-forecasting

import tensorflow as tf
from tensorflow.keras.layers import Dense
from ..layers.wavenet_layer import Dense3D, TemporalConv


params = {
    'dilation_rates': [2 ** i for i in range(4)],
    'kernel_sizes': [2 for i in range(4)],
    'filters': 128,
    'dense_hidden_size': 64
}


class WaveNet(object):
    """WaveNet network

    """
    def __init__(self, custom_model_params={}, dynamic_decoding=True):
        params.update(custom_model_params)
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def __call__(self, inputs, training, predict_seq_length, teacher=None):
        if isinstance(inputs, tuple):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:  # for single variable prediction
            encoder_feature = x = inputs
            decoder_feature = None

        encoder_output, encoder_states = self.encoder(encoder_feature)
        decoder_output = self.decoder(x,
                                      decoder_feature,
                                      encoder_states=encoder_states,
                                      predict_seq_length=predict_seq_length,
                                      teacher=teacher)
        return decoder_output


class Encoder(object):
    def __init__(self, params):
        self.params = params
        self.conv_times = []
        for i, (kernel_size, dilation) in enumerate(zip(self.params['kernel_sizes'], self.params['dilation_rates'])):
            self.conv_times.append(TemporalConv(filters=2 * self.params['filters'],
                                                kernel_size=kernel_size,
                                                causal=True,
                                                dilation_rate=dilation))
        self.dense_time1 = Dense3D(units=self.params['filters'], activation='tanh', name='encoder_dense_time1')
        self.dense_time2 = Dense3D(units=self.params['filters'] + self.params['filters'], name='encoder_dense_time2')
        self.dense_time3 = Dense3D(units=self.params['dense_hidden_size'], activation='relu', name='encoder_dense_time3')
        self.dense_time4 = Dense3D(units=1, name='encoder_dense_time_4')

    def forward(self, x):
        """
        :param x:
        :return: conv_inputs [batch_size, time_sequence_length, filters] * time_sequence_length
        """
        inputs = self.dense_time1(inputs=x)  # batch_size * time_sequence_length * filters

        skip_outputs = []
        conv_inputs = [inputs]
        for conv_time in self.conv_times:
            dilated_conv = conv_time(inputs)
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
            outputs = self.dense_time2(inputs=dilated_conv)
            skips, residuals = tf.split(outputs, [self.params['filters'], self.params['filters']], axis=2)
            inputs += residuals
            conv_inputs.append(inputs)  # batch_size * time_sequence_length * filters
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = self.dense_time3(skip_outputs)
        y_hat = self.dense_time4(h)
        return y_hat, conv_inputs[:-1]

    def __call__(self, x):
        return self.forward(x)


class Decoder(object):
    def __init__(self, params):
        self.params = params
        self.dense_1 = Dense(self.params['filters'], activation='tanh', name='decoder_dense_1')
        self.dense_2 = Dense(2 * self.params['filters'], name='decoder_dense_2')
        self.dense_3 = Dense(2 * self.params['filters'], use_bias=False, name='decoder_dense_3')
        self.dense_4 = Dense(2 * self.params['filters'], name='decoder_dense_4')
        self.dense_5 = Dense(self.params['dense_hidden_size'], activation='relu', name='decoder_dense_5')
        self.dense_6 = Dense(1, name='decoder_dense_6')

    def foward(self, x, decoder_feature, encoder_states, predict_seq_length, teacher):
        decoder_init_value = x[:, -1, :]

        def cond_fn(time, prev_output, decoder_output_ta):
            return time < predict_seq_length

        def body(time, prev_output, decoder_output_ta):
            if time == 0 or teacher is None:
                current_input = prev_output
            else:
                current_input = teacher[:, time - 1, :]

            if decoder_feature is not None:
                current_feature = decoder_feature[:, time, :]
                current_input = tf.concat([current_input, current_feature], axis=1)

            inputs = self.dense_1(current_input)

            skip_outputs = []
            for i, dilation in enumerate(self.params['dilation_rates']):
                state = encoder_states[i][:, -dilation, :]

                dilated_conv = self.dense_2(state) + self.dense_3(inputs)
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
                outputs = self.dense_4(dilated_conv)
                skips, residuals = tf.split(outputs, [self.params['filters'], self.params['filters']], axis=1)
                inputs += residuals
                encoder_states[i] = tf.concat([encoder_states[i], tf.expand_dims(inputs, 1)], axis=1)
                skip_outputs.append(skips)

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            h = self.dense_5(skip_outputs)
            y_hat = self.dense_6(h)
            decoder_output_ta = decoder_output_ta.write(time, y_hat)
            return time + 1, y_hat, decoder_output_ta

        loop_init = [tf.constant(0, dtype=tf.int32),
                     decoder_init_value,
                     tf.TensorArray(dtype=tf.float32, size=predict_seq_length)]
        _, _, decoder_outputs_ta = tf.while_loop(cond=cond_fn, body=body, loop_vars=loop_init)
        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs

    def __call__(self, x, decoder_feature, encoder_states, predict_seq_length, teacher=None):
        return self.foward(x,
                           decoder_feature,
                           encoder_states,
                           predict_seq_length=predict_seq_length,
                           teacher=teacher)


class Decoder2(object):
    def __init__(self, predict_sequence_length=24):
        self.predict_sequence_length = predict_sequence_length
        self.dilation_rates = dilation_rates
        self.dense1 = Dense(filters, activation='tanh')
        self.dense2 = Dense(2 * filters, use_bias=True)
        self.dense3 = Dense(2 * filters, use_bias=False)
        self.dense4 = Dense(2 * filters)
        self.dense5 = Dense(dense_hidden_size, activation='relu')
        self.dense6 = Dense(1)

    def __call__(self, decoder_feature, encoder_feature, encoder_states, teacher=None):
        this_output = encoder_feature[:, -1, 0:1]  # 预测变量的历史值存放在第一位
        decoder_outputs = []

        for i in range(self.predict_sequence_length):
            if teacher is None:
                this_input = tf.concat([this_output, decoder_feature[:, i, :]], axis=-1)  # batch * 2
            else:
                this_input = tf.concat([teacher[:, i:i+1], decoder_feature[:, i%12:i%12+1]], axis=-1)

            x = self.dense1(this_input)
            skip_outputs = []

            for i, dilation in enumerate(self.dilation_rates):
                state = encoder_states[i][:, -dilation, :]

                # use 2 dense layer to calculate a kernel=2 convlution
                dilated_conv = self.dense2(state) + self.dense3(x)
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
                out = self.dense4(dilated_conv)
                skip, residual = tf.split(out, 2, axis=1)
                x += residual
                #x = residual
                # while new time state generated, extend it to states
                encoder_states[i] = tf.concat([encoder_states[i], tf.expand_dims(x, 1)], axis=1)
                skip_outputs.append(skip)

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            skip_outputs = self.dense5(skip_outputs)
            this_output = self.dense6(skip_outputs)
            decoder_outputs.append(this_output)

        decoder_outputs = tf.concat(decoder_outputs, axis=1)
        return decoder_outputs
