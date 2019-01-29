import models.weibull
import models.LSTM
from prepare_data import Prepare_Denza,Prepare_Mercedes
import prepare_model_input

class Config:
    pass

def main():

    denza_400 = Prepare_Denza(interval_type='monthly',
                              production_file='./raw_data/production_2018_400.xlsx',
                              repair_file='./raw_data/failure_2018_400.xlsx',
                              output_file='Weibull_data_400.csv')

main()
