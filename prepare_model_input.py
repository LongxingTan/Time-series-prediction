import numpy as np
import pandas as pd
import os

class Input_builder(object):
    def __init__(self,data_dir):
        self._read_csv(os.path.join('./data',data_dir))

    def create_weibull_input(self):
        x, y = self.examples['Interval'].values, self.examples['Failure_rate_cum'].values
        return x,y


    def create_RNN_input(self,time_state):
        data=self.examples.iloc[:,-1].values
        x,y=[],[]
        for i in range(len(data)-time_state-1):
            x.append(data[i:i+time_state])
            y.append(data[i+time_state])
        x=np.array(x)
        x= x.reshape(x.shape[0],x.shape[1],1)
        y=np.array(y)
        y=y.reshape(y.shape[0],1)
        return x,y

    def create_seq2seq_input(self,input_seq_length,output_seq_length):
        data=self.examples.iloc[:,-1].values
        x,y=[],[]
        for i in range(len(data)-input_seq_length-output_seq_length-1):
            x.append([data[i:(i+input_seq_length)]])
            y.append([data[(i+input_seq_length):(i+input_seq_length+output_seq_length)]])

        x = np.array(x)
        x2 = x.reshape(x.shape[0],-1, x.shape[1])
        y= np.array(y)
        y2 = y.reshape(y.shape[0],-1,y.shape[1])
        return x2,y2

    def _read_csv(self,data_dir):
        self.examples=pd.read_csv(data_dir)