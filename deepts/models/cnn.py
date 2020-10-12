# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-05
# paper:
# other implementations: https://github.com/LenzDu/Kaggle-Competition-Favorita/blob/master/cnn.py
#                        https://github.com/philipperemy/keras-tcn
#                        https://github.com/emreaksan/stcn


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
from deepts.layers.wavenet_layer import Dense3D, ConvTime


params={
    'dilation_rates':[2 ** i for i in range(4)],
    'kernel_sizes':[2 for i in range(4)],
    'filters':128,
    'dense_hidden_size':64
}


class CNN(object):
    def __init__(self, custom_model_params={}):
        self.params = params
        self.conv_times = []
        for i, (dilation, kernel_size) in enumerate(zip(self.params['dilation_rates'], self.params['kernel_sizes'])):
            self.conv_times.append(ConvTime(filters=2 * self.params['filters'],
                                            kernel_size=kernel_size,
                                            causal=True,
                                            dilation_rate=dilation))
        self.dense_time1 = Dense3D(units=self.params['filters'], name='encoder_dense_time_1')
        self.dense_time2 = Dense3D(units=self.params['filters'] + self.params['filters'], name='encoder_dense_time_2')
        self.dense_time3 = Dense3D(units=1, name='encoder_dense_time_3')

    def __call__(self, x):
        input=x

        c1 = self.conv_times[0](x)
        c2 = self.conv_times[1](c1)
        c2 = self.conv_times[2](c2)
        c2 = self.conv_times[3](c2)
        print(c2.shape)

        c4 = tf.concat([c1, c2], axis=-1)
        conv_out = Conv1D(8, 1, activation='relu')(c4)
        conv_out = Dropout(0.25)(conv_out)
        # conv_out=Flatten()(conv_out)

        conv_out = self.dense_time3(conv_out)
        print(conv_out.shape)
        return conv_out
