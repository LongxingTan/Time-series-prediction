# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-03

import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import numpy as np
import matplotlib.pyplot as plt
from data.data_reader import PassengerData
from deepts.model import Model
from config import params


def main(plot=False):
    x,y=PassengerData(params).get_examples(data_dir='../data/international-airline-passengers.csv',sample=0.2)
    print(x.shape,y.shape)

    model=Model(params=params,use_model=params['use_model'])
    y_pred=model.predict(x.astype(np.float32), model_dir=params['saved_model_dir'])
    print(y_pred)

    if plot:
        for i in range(y_pred.shape[1]):
            plt.subplot(y_pred.shape[1],1,i+1)
            plt.plot(y[:,i,0],label='true')
            plt.plot(y_pred[:,i],label='pred')
            plt.legend()
        plt.show()

        for i in range(36):
            plt.subplot(6,6,i+1)
            i=np.random.choice(range(y_pred.shape[0]))
            plt.plot(y[i,:,0],label='true')
            plt.plot(y_pred[i,:],label='pred')
            plt.legend()
        plt.show()

    return y,y_pred


if __name__=='__main__':
    main(plot=True)
