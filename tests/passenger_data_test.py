
import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

from examples.data import data_reader

data=data_reader.PassengerData(data_dir='../data/international-airline-passengers.csv')
data.get_examples()