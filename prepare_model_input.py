import numpy as np
import pandas as pd

class Input_builder(object):
    def __init__(self,data_dir):
        self._read_csv(data_dir)

    def create_weibull_input(self):
        x, y = self.examples['Interval'].values, self.examples['Failure_rate_cum'].values
        return x,y


    def create_RNN_input(self,time_state=6):
        data=self.examples
        x,y=[],[]
        for i in range(len(data)-time_state-1):
            x.append(data[i:i+time_state,-1])
            y.append(data[i+time_state,-1])
        x=np.array(x)
        x= x.reshape(x.shape[0],x.shape[1],1)
        y=np.array(y)
        y=y.reshape(y.shape[0],1)
        return x,y

    def create_seq2seq_input(self,input_seq_length,output_seq_length):
        data=self.examples
        x,y=[],[]
        for i in range(len(data)-input_seq_length-output_seq_length-1):
            x.append([data[i:(i+input_seq_length)]])
            y.append([data[(i+input_seq_length):(i+input_seq_length+output_seq_length)]])
        return x,y

    def _read_csv(self,data_dir):
        self.examples=pd.read_csv(data_dir)

