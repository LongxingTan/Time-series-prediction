import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Input_builder(object):
    def __init__(self):
        pass

    def __call__(self, models,x,y=None,train_window=20,train_window_2=None):
        if models=='weibull':
            return self.create_weibull_input(x,y,train_window)
        elif models=='svm' or models=='lstm':
            return self.create_RNN_input(x,train_window=20)
        elif models=='seq2seq':
            return self.create_seq2seq_basic_input(x,train_window,train_window_2)

    def create_weibull_input(self,x,y,train_windows=20):
        index_end=len(y)-1
        y=list(y)
        for yy in y[::-1]:
            if yy!=y[-1]:
                index_end=y.index(yy)
                break
        index_begin=index_end-train_windows if (index_end-train_windows>0) else 1
        x,y=x[index_begin:index_end],y[index_begin:index_end]
        logging.info("Weibull train data {}".format(len(x)))
        return np.array(x),np.array(y)

    def create_RNN_input(self,x_train,train_window):
        #data=self.examples.iloc[:,-1].values
        x,y=[],[]
        for i in range(len(x_train)-train_window-1):
            x.append(x_train[i:i+train_window])
            y.append(x_train[i+train_window])
        x=np.array(x)
        x= x.reshape(x.shape[0],x.shape[1],1)
        y=np.array(y)
        y=y.reshape(y.shape[0],1)
        return x,y

    def create_seq2seq_basic_input(self,data,input_seq_length,output_seq_length):
        #data=self.examples.iloc[:,-1].values
        x,y=[],[]
        for i in range(len(data)-input_seq_length-output_seq_length-1):
            x.append([data[i:(i+input_seq_length)]])
            y.append([data[(i+input_seq_length):(i+input_seq_length+output_seq_length)]])

        x = np.array(x)
        x2 = x.reshape(x.shape[0],-1, x.shape[1])
        y= np.array(y)
        y2 = y.reshape(y.shape[0],-1,y.shape[1])
        return x2,y2

    def create_seq2seq_input(self):
        pass

    def create_arima_input(self,examples):
        data = examples.iloc[:,-1].values
        return data

    def _read_csv(self,data_dir):
        examples=pd.read_csv(data_dir)
        return examples

    def _normalize(self,data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(data)


class Input_pipe(object):
    def __init__(self):
        pass

    def get_train_features(self):
        pass

    def get_dev_features(self):
        pass

    def get_test_features(self):
        pass

    def create_examples2features(self):
        pass