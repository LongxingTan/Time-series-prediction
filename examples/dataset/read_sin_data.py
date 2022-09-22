# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# this script shows several examples reading the data, next step is load_data.py

import os
import logging
import random
import numpy as np
import pandas as pd
# import modin.pandas as pd   # faster Pandas by Berkeley
import matplotlib.pyplot as plt
from .features import transform2_lagged_feature, multi_step_y
from .norm_feature import FeatureNorm


class DataSet(object):
    def __init__(self, params):
        self.params = params

    def load_data(self, data_dir, *args):
        raise NotImplementedError('must implement load_data')

    def get_examples(self, *args):
        raise NotImplementedError('must implement get_examples')

    def save_tf_record(self, *args):
        pass


class SineData(DataSet):
    def __init__(self,params):
        '''
        This is a toy data example of sine function with some noise
        :param params:
        '''
        super(SineData, self).__init__(params)

    def load_data(self, data_dir=None, *args):
        n_examples = 1000
        sequence_length = 30
        predict_sequence_length = 10

        x = []
        y = []
        for _ in range(n_examples):
            rand = random.random()*2*np.pi
            sig1 = np.sin(np.linspace(rand, 3.*np.pi+rand, sequence_length + predict_sequence_length))
            sig2 = np.cos(np.linspace(rand, 3.*np.pi+rand, sequence_length + predict_sequence_length))

            x1 = sig1[:sequence_length]
            y1 = sig1[sequence_length:]
            x2 = sig2[:sequence_length]
            y2 = sig2[sequence_length:]

            x_ = np.array([x1, x2])
            y_ = np.array([y1, y2])

            x.append(x_.T)
            y.append(y_.T)

        x = np.array(x)
        y = np.array(y)
        return x, y

    def get_examples(self, data_dir=None, sample=1, plot=False):
        x, y = self.load_data()

        if plot:
            plt.plot(x[0, :])
            plt.show()
        return x, y[:, :, 0:1]


class PassengerData(DataSet):
    def __init__(self, params):
        '''
        This is a simple uni-variable time series data set: airline passenger data
        :param data_dir:
        :param params:
        '''
        super(PassengerData, self).__init__(params)
        self.feature_norm = FeatureNorm()

    def preprocess_data(self, data):
        data.columns = ['Month', 'Passengers']
        return data

    def load_data(self, data_dir, skipfooter=0, parse_dates=None, date_parser=None):
        '''
        Use pandas to load the data from data_dir
        :param data_dir: the directory of the processed time series file
        :return: dataframe of the data
        '''
        if not os.path.exists(data_dir):
            raise ValueError("The file {} does not exists".format(data_dir))

        data = pd.read_csv(data_dir, parse_dates=parse_dates, date_parser=date_parser, skipfooter=skipfooter)
        logging.info("input data shape : {}".format(data.shape))
        return data

    def get_examples(self, data_dir, sample=1, start_date=None, plot=False, model_dir='../weights'):
        data = self.load_data(data_dir)
        data = self.preprocess_data(data)

        x = pd.DataFrame(data['Passengers'])
        x = self.feature_norm(x, model_dir=model_dir)
        x = transform2_lagged_feature(x, window_sizes=self.params['input_seq_length'])
        y = pd.DataFrame(data['Passengers'])
        y = np.log1p(y.values)
        y = multi_step_y(y, predict_window=self.params['output_seq_length'])
        x, y = self.postprocess(x, y)

        if plot:
            plt.plot(data['Passengers'])
            plt.show()
            in_len = self.params['input_seq_length']
            total_len = in_len + self.params['output_seq_length']
            plt.plot(range(0, in_len), x[40, :, 0])
            plt.plot(range(in_len, total_len), y[40, :, 0])
            plt.show()

        n_example = int(sample * x.shape[0])
        if sample >= 0.5:  # if sample>0.5, blindly guess it's training
            print('x:', x[:n_example].shape, ' y:', y[:n_example].shape)
            return x[:n_example], y[:n_example]
        if sample < 0.5:  # if sample<0.5, blindly guess it's valid
            print('x:', x[n_example:].shape, ' y:', y[n_example:].shape)
            return x[n_example:], y[n_example:]
        #for x1,y1 in zip(x,y):
            #yield x1,y1

    def postprocess(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if len(x.shape) == 2:
            x = x[..., np.newaxis]
        if isinstance(y, pd.DataFrame):
            y = y.values
        if len(y.shape) == 2:
            y = y[..., np.newaxis]

        filter = (np.isnan(y)).any(axis=1)[:, 0]  # remove the nan in target
        x = x[~filter]
        y = y[~filter]
        return x, y


if __name__ == '__main__':
    # data_dir = '../../data/international-airline-passengers.csv'
    #
    # prepare_data = PassengerData(params={'input_seq_length':15,'output_seq_length':5})
    # x,y=prepare_data.get_examples(data_dir,sample=0.8,plot=True,model_dir='../../models')
    # print(x.shape,y.shape)

    data_reader=SineData(params={})
    x, y = data_reader.get_examples(sample=1, plot=True)
    print(x.shape, y.shape)
