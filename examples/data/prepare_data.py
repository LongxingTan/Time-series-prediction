# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import os
import numpy as np
import pandas as pd
#import modin.pandas as pd
import logging
from feature.prepare_feature import transform2_lagged_feature,multi_step_y
from feature.norm_feature import FeatureNorm


class DataSet(object):
    def __init__(self,params):
        self.params = params

    def load_data(self,data_dir,*args):
        raise NotImplementedError('must implement load_data')

    def preprocess_data(self,*args):
        raise NotImplementedError('must implement preprocess_data')

    def get_examples(self,*args):
        raise NotImplementedError('must implement get_examples')

    def save_tf_record(self,*args):
        pass


class PassengerData(DataSet):
    def __init__(self,params):
        '''
        For airline passenger data, refer to the notebook/data_introduction.ipynb
        :param data_dir:
        :param params:
        '''
        super(PassengerData,self).__init__(params)
        self.feature_norm=FeatureNorm()

    def preprocess_data(self,data):
        data.columns = ['Month', 'Passengers']
        return data

    def load_data(self,data_dir, skipfooter=0, engine='python', parse_dates=None, date_parser=None):
        '''
        Use pandas to load the data from data_dir
        :param data_dir: the directory of the processed time series file
        :return: dataframe of the data
        '''
        if not os.path.exists(data_dir):
            raise ValueError("The file {} does not exists".format(data_dir))

        data = pd.read_csv(data_dir, parse_dates=parse_dates, date_parser=date_parser, skipfooter=skipfooter,engine=engine)
        logging.info("input data shape : {}".format(data.shape))
        return data

    def get_examples(self,data_dir):
        data=self.load_data(data_dir)
        data=self.preprocess_data(data)

        x = pd.DataFrame(data['Passengers'])
        x = self.feature_norm(x)
        x = transform2_lagged_feature(x)
        y = pd.DataFrame(data['Passengers'])
        y = np.log1p(y.values)
        y = multi_step_y(y,5)
        x,y=self.postprocess(x,y)
        #for x1,y1 in zip(x,y):
            #yield x1,y1
        return (x,y)

    def postprocess(self,x,y):
        if isinstance(x,pd.DataFrame):
            x=x.values
        if len(x.shape)==2:
            x=x[...,np.newaxis]
        if isinstance(y,pd.DataFrame):
            y=y.values
        if len(y.shape)==2:
            y=y[...,np.newaxis]

        filter=(np.isnan(y)).any(axis=1)[:,0]  # remove the nan in target
        x=x[~filter]
        y=y[~filter]
        return x,y
