

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from deepts.layers.tcn_layer import Dense3D, ConvTime
tf.config.experimental_run_functions_eagerly(True)  # ??


# https://github.com/philipperemy/keras-tcn
# https://github.com/emreaksan/stcn


params={
    'dilation_rates':[2 ** i for i in range(4)],
    'kernel_sizes':[2 for i in range(4)],
    'filters':128,
    'dense_hidden_size':64,
    'predict_window_sizes':5,
}


class TCN(object):
    '''
    Temporal convolutional network
    '''
    def __init__(self):
        self.encoder=Encoder(params)
        self.decoder=Decoder(params)

    def __call__(self, inputs_shape):
        x=Input(inputs_shape)
        encoder_outputs,encoder_state=self.encoder(x)
        decoder_output = self.decoder(None,encoder_outputs=encoder_outputs,encoder_inputs=x)
        return tf.keras.Model(x,decoder_output)


class Encoder(object):
    def __init__(self,params):
        self.params=params
        self.conv_times=[]
        for i, (dilation, kernel_size) in enumerate(zip(self.params['dilation_rates'], self.params['kernel_sizes'])):
            self.conv_times.append(ConvTime(filters=2 * self.params['filters'],
                                            kernel_size=kernel_size,
                                            causal=True,
                                            dilation_rate=dilation))
        self.dense_time1=Dense3D(units=self.params['filters'],name='encoder_dense_time_1')
        self.dense_time2=Dense3D(units=self.params['filters'] + self.params['filters'],name='encoder_dense_time_2')
        self.dense_time3=Dense3D(units=self.params['dense_hidden_size'],name='encoder_dense_time_3')

    def forward(self,x):
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
        h = tf.nn.relu(self.dense_time3(skip_outputs))
        return conv_inputs[:-1], h

    def __call__(self, x):
        return self.forward(x)


class Decoder(object):
    def __init__(self,params):
        self.params=params
        self.dense_time_1=Dense3D(1,name='decoder_dense_time_1')
        self.dense_1=tf.keras.layers.Dense(self.params['filters'],activation='tanh',name='decoder_dense_1')
        self.dense_2=tf.keras.layers.Dense(2 * self.params['filters'],name='decoder_dense_2')
        self.dense_3=tf.keras.layers.Dense(2 * self.params['filters'],name='decoder_dense_3')
        self.dense_4=tf.keras.layers.Dense(2 * self.params['filters'],name='decoder_dense_4')
        self.dense_5=tf.keras.layers.Dense(self.params['dense_hidden_size'],name='decoder_dense_5')
        self.dense_6=tf.keras.layers.Dense(1,name='decoder_dense_6')

    def foward(self,decoder_inputs,encoder_outputs,decoder_init_value):
        def cond_fn(time, prev_output, decoder_output_ta):
            return time < self.params['predict_window_sizes']

        def body(time, prev_output, decoder_output_ta):
            current_input=prev_output
            if decoder_inputs is not None:
                current_feature = decoder_inputs[:, time, :]
                current_input = tf.concat([current_input, current_feature], axis=1)

            inputs = self.dense_1(current_input)

            skip_outputs, conv_inputs = [], []
            for i, dilation in enumerate(self.params['dilation_rates']):
                encoder_output=encoder_outputs[i]
                state=self.dense_time_1(encoder_output)
                state=tf.squeeze(state,2)

                dilated_conv=self.dense_2(state)+self.dense_3(inputs)
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
                outputs=self.dense_4(dilated_conv)
                skips, residuals = tf.split(outputs, [self.params['filters'], self.params['filters']], axis=1)
                inputs += residuals
                conv_inputs.append(inputs)
                skip_outputs.append(skips)

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            h=self.dense_5(skip_outputs)
            y_hat=self.dense_6(h)
            decoder_output_ta.write(time,y_hat)
            return time + 1, y_hat, decoder_output_ta

        loop_init = [tf.constant(0, dtype=tf.int32),
                     decoder_init_value,
                     tf.TensorArray(dtype=tf.float32, size=self.params['predict_window_sizes'])]
        _, _, decoder_outputs_ta = tf.while_loop(cond=cond_fn, body=body, loop_vars=loop_init)
        decoder_outputs = decoder_outputs_ta.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs

    def __call__(self, decoder_inputs,encoder_outputs,encoder_inputs):
        return self.foward(decoder_inputs,
                           encoder_outputs,
                           decoder_init_value=tf.expand_dims(encoder_inputs[:,-1,0],1))
