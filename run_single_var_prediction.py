
import os
from models.arima import Time_ARIMA
from models.xgb import Time_XGB
from models.svm import Time_SVM
from create_model_input import Input_builder
from load_data import Load_Mercedes
from config import params
import pandas as pd


def run_pred(params,models,mode='predict'):
    model = models(params)
    if mode == 'train' and os.path.exists(params['calendar_data']):
        pass
    else:
        Load_Mercedes(output_date_type='calendar', interval_type=params['interval_type'],
                      production_file=params['production_file'],
                      repair_file=params['failure_file'], )
    examples = pd.read_csv(params['calendar_data'])
    examples=examples.iloc[:,-1].values

    if mode=='train':
        split=int(len(examples)*0.7)
        train_data,test_data=examples[:split],examples[split:]
        train_data,test_data=Input_builder()(models=models)
        model.train(train_data)
        predictions=model.predict_point(train_data,test_data,predict_window=len(test_data))
        model.plot(train=train_data,test=test_data,predictions=predictions)

    elif mode=='predict':
        train_data=examples
        model.train(train_data)


if __name__=="__main__":
    #run_pred(params,models=Time_ARIMA,mode='train')
    run_pred(params,models=Time_XGB,mode='train')
