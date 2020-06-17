
import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import pytest
from examples.data import read_data


data_reader=read_data.SineData(params={'input_seq_length':15,'output_seq_length':5})
x,y=data_reader.get_examples(plot=True)
