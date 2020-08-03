# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper:
# other implementations: https://github.com/sjvasquez/web-traffic-forecasting


import tensorflow as tf
from tensorflow.keras.layers import Dense
from deepts.layers.wavenet_layer import Dense3D, ConvTime


params = {
    'dilation_rates': [2 ** i for i in range(4)],
    'kernel_sizes': [2 for i in range(4)],
    'filters': 128,
    'dense_hidden_size': 64
}


class WaveNet(object):
    '''
    Temporal convolutional network
    '''
    def __init__(self, custom_model_params={}):
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
        decoder_output = self.decoder(x, decoder_feature, encoder_outputs=encoder_states, predict_seq_length=predict_seq_length, teacher=teacher)
        return decoder_output


class Encoder(object):
    def __init__(self, params):
        self.params = params
        self.conv_times = []
        for i, (dilation, kernel_size) in enumerate(zip(self.params['dilation_rates'], self.params['kernel_sizes'])):
            self.conv_times.append(ConvTime(filters=2 * self.params['filters'],
                                            kernel_size=kernel_size,
                                            causal=True,
                                            dilation_rate=dilation))
        self.dense_time1 = Dense3D(units=self.params['filters'], activation='tanh', name='encoder_dense_time_1')
        self.dense_time2 = Dense3D(units=self.params['filters'] + self.params['filters'], name='encoder_dense_time_2')
        self.dense_time3 = Dense3D(units=self.params['dense_hidden_size'], activation='relu', name='encoder_dense_time_3')
        self.dense_time4 = Dense3D(units=1, name='encoder_dense_time_4')

    def forward(self, x):
        '''
        :param x:
        :return: conv_inputs [batch_size, time_sequence_length, filters] * time_sequence_length
        '''
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
        self.dense_time_1 = Dense3D(1, name='decoder_dense_time_1')
        self.dense_1 = tf.keras.layers.Dense(self.params['filters'], activation='tanh', name='decoder_dense_1')
        self.dense_2 = tf.keras.layers.Dense(2 * self.params['filters'], name='decoder_dense_2')
        self.dense_3 = tf.keras.layers.Dense(2 * self.params['filters'], name='decoder_dense_3')
        self.dense_4 = tf.keras.layers.Dense(2 * self.params['filters'], name='decoder_dense_4')
        self.dense_5 = tf.keras.layers.Dense(self.params['dense_hidden_size'], name='decoder_dense_5')
        self.dense_6 = tf.keras.layers.Dense(1, name='decoder_dense_6')

    def foward(self,x, decoder_feature, encoder_outputs, predict_seq_length, teacher):
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

            skip_outputs, conv_inputs = [], []
            for i, dilation in enumerate(self.params['dilation_rates']):
                state = encoder_outputs[i][:, -dilation, :]

                dilated_conv = self.dense_2(state) + self.dense_3(inputs)
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
                outputs = self.dense_4(dilated_conv)
                skips, residuals = tf.split(outputs, [self.params['filters'], self.params['filters']], axis=1)
                inputs += residuals
                encoder_outputs[i] = tf.concat([encoder_outputs[i], tf.expand_dims(inputs, 1)], axis=1)
                skip_outputs.append(skips)

                # encoder_output=encoder_outputs[i]
                # state=self.dense_time_1(encoder_output)
                # state=tf.squeeze(state,2)
                #
                # dilated_conv=self.dense_2(state)+self.dense_3(inputs)
                # conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                # dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
                # outputs=self.dense_4(dilated_conv)
                # skips, residuals = tf.split(outputs, [self.params['filters'], self.params['filters']], axis=1)
                # inputs += residuals
                # conv_inputs.append(inputs)
                # skip_outputs.append(skips)

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            h = self.dense_5(skip_outputs)
            y_hat = self.dense_6(h)
            decoder_output_ta.write(time, y_hat)
            return time + 1, y_hat, decoder_output_ta

        loop_init = [tf.constant(0, dtype=tf.int32),
                     decoder_init_value,
                     tf.TensorArray(dtype=tf.float32, size=predict_seq_length)]
        _, _, decoder_outputs_ta = tf.while_loop(cond=cond_fn, body=body, loop_vars=loop_init)
        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs

    def __call__(self, x, decoder_feature, encoder_outputs, predict_seq_length, teacher=None):
        return self.foward(x,
                           decoder_feature,
                           encoder_outputs,
                           predict_seq_length=predict_seq_length,
                           teacher=teacher)
