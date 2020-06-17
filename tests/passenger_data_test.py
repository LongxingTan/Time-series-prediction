
import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import pytest
from examples.data import read_data


data_dir = '../data/international-airline-passengers.csv'
prepare_data = read_data.PassengerData(params={'input_seq_length':15,'output_seq_length':5})
x,y=prepare_data.get_examples(data_dir,sample=0.8,plot=True,model_dir='../models')
print(x.shape,y.shape)
