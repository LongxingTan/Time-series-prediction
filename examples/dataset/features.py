# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def transform2_lagged_feature(x, window_sizes):
    '''
    create historical lagged value as features
    :return:
    '''
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x.values, index=range(len(x)), columns=["Feature"])
    inputs_lagged = pd.DataFrame()
    init_value = x.iloc[0]
    for window_size in range(1, window_sizes + 1):
        inputs_roll = np.roll(x, window_size, axis=0)
        inputs_roll[:window_size] = init_value
        inputs_roll = pd.DataFrame(inputs_roll, index=x.index,
                                   columns=[i + '_lag{}'.format(window_size) for i in x.columns])
        inputs_lagged = pd.concat([inputs_roll, inputs_lagged], axis=1)
    return inputs_lagged


def multi_step_y(y, predict_window, predict_gap=1):
    if isinstance(y, pd.DataFrame):
        y = y.values
    outputs = np.full((y.shape[0], predict_window), np.nan)
    y = y[:, 0]
    for i in range(predict_window):
        outputs_column = np.roll(y, -(i+predict_gap-1)).astype(np.float)
        if (i+predict_gap-1) != 0:
            outputs_column[-(i+predict_gap-1):] = np.nan
        outputs[:, i] = outputs_column
    return outputs


def simple_moving_average(x):
    pass




class FeatureNorm(object):
    def __init__(self, type='minmax'):
        self.type = type

    def __call__(self, x, mode='train', model_dir='../weights', name='scaler'):
        assert len(x.shape) == 2, "Input rank for FeatureNorm should be 2"
        if self.type == 'standard':
            scaler = StandardScaler()
        elif self.type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported norm type yet: {}".format(self.type))

        if mode == 'train':
            scaler.fit(x)
            joblib.dump(scaler, os.path.join(model_dir, name+'.pkl'))
        else:
            scaler = joblib.load(os.path.join(model_dir, name+'.pkl'))
        output = scaler.transform(x)
        try:
            return pd.DataFrame(output, index=x.index, columns=x.columns)
        except:
            return output
