#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import tensorflow as tf


class Loss(object):
    def __init__(self, use_loss):
        self.use_loss = use_loss

    def __call__(self,):
        if self.use_loss == 'mse':
            return tf.keras.losses.MeanSquaredError()
        elif self.use_loss == 'rmse':
            return tf.math.sqrt(tf.keras.losses.MeanSquaredError())
        elif self.use_loss == 'huber':
            return tf.keras.losses.Huber(delta=1.0)
        elif self.use_loss == 'gaussian_likelihood':
            return Gaussian()
        else:
            raise ValueError("Not supported use_loss yet: {}".format(self.use_loss))


class Gaussian(object):
    def __init__(self, sigma=0):
        # in order to use a general framework, so it's not used here, but in y_pred
        self.sigma = sigma

    def __call__(self, y_true, y_pred):
        y_pred, self.sigma = y_pred
        loss = tf.reduce_mean(0.5 * tf.math.log(self.sigma) +
                              0.5 * tf.math.truediv(tf.math.square(y_true - y_pred), self.sigma)) + 1e-7 + 6
        return loss
