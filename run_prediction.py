
import os
from models.arima import Time_ARIMA
from models.xgb import Time_XGB
from models.svm import Time_SVM
from models._ma import moving_avg
from create_model_input import Input_builder
from load_data import Load_Mercedes
from config import params
import pandas as pd


def run_pred(params,models,mode='predict',grid_search=False,ma_window=1):
    model = models(params)
    if mode == 'train' and os.path.exists(params['calendar_data']):
        pass
    else:
        Load_Mercedes(output_date_type='calendar', interval_type=params['interval_type'],
                      production_file=params['production_file'],
                      repair_file=params['failure_file'], )
    examples = pd.read_csv(params['calendar_data'])
    examples=moving_avg(examples,window_size=ma_window)

    if mode=='train':
        split=int(len(examples)*0.7)
        train_data,test_data=examples.iloc[:split,:],examples.iloc[split:,:]

        train_data=Input_builder()(model=model,x=train_data)
        test_data=Input_builder()(model=model,x=test_data)
        model.train(train_data,grid_search=grid_search)
        predictions,pred_low,pred_high=model.predict(test=test_data,predict_window=len(test_data))
        model.plot(train_data,test=test_data, predictions=predictions, predictions_low=pred_low, predictions_high=pred_high)

    elif mode=='predict':
        train_data=examples
        model.train(train_data,grid_search=grid_search)
        predictions, pred_low, pred_high = model.predict(train=pd.DataFrame(train_data), predict_window=10)
        model.plot(train_data, predictions=predictions, predictions_low=pred_low, predictions_high=pred_high)


if __name__=="__main__":
    #run_pred(params,models=Time_ARIMA,mode='train',grid_search=True,ma_window=4)
    run_pred(params,models=Time_XGB,mode='train',grid_search=True,ma_window=1)

