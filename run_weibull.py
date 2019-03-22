# Weibull is a common used survival model
__author__ = 'LongxingTan'

import os
import pandas as pd
import models.weibull
from load_data import Load_Mercedes


class Config(object):
    interval_type = 'weekly'
    production_file = './raw_data/samples'
    failure_file = './raw_data/failures'
    interval_data='./data/interval_data.csv'
    calendar_data='./data/calendar_data.csv'

def run_prediction(config,mode='prediction'):
    if mode=='train' and os.path.exists(config.interval_data):
        examples=data_aggby_interval=pd.read_csv(config.interval_data)
        failures_aggby_calendar=pd.read_csv(config.calendar_data)
    else:
        mercedes = Load_Mercedes(output_date_type='interval',
                                 interval_type=config.interval_type,
                                 production_file=config.production_file,
                                 repair_file=config.failure_file, )
        examples=mercedes.data_aggby_interval

    x,y=examples['Interval'].values, examples['Failure_rate_cum'].values
    weibull_model = models.weibull.Weibull_model()

    if mode=='train':
        split=int(0.7*len(x))
        train_x,test_x=x[:split],x[split:]
        train_y,test_y=y[:split],y[split:]

        weibull_model.train(train_x,train_y,train_window=50)
        x_full_pred,y_full_pred=weibull_model.predict_by_interval(predicted_interval=len(test_x),return_full=False)
        weibull_model.plot_two(train_x,train_y,x_full_pred,y_full_pred)

        calerdar_split=int(len(failures_aggby_calendar)*0.7)
        failures_calendar=weibull_model.predict_by_calendar(interval_samples=data_aggby_interval.ix[:split-1,'Samples'],
                                                            calendar_failures=failures_aggby_calendar.iloc[:calerdar_split,:],
                                                            predicted_interval=len(failures_aggby_calendar)-calerdar_split,
                                                            output_file='./result/calendar_predict.csv',)
        weibull_model.plot_calendar(failures_predict=failures_calendar,further_true=failures_aggby_calendar.iloc[calerdar_split:,:])

    elif mode=='predict':
        weibull_model.train(x,y,train_window=40)
        weibull_model.plot()

        weibull_model.predict_by_calendar(interval_samples=mercedes.data_aggby_interval['Samples'],
                                          calendar_failures=mercedes.failures_aggby_calendar,
                                          predicted_interval=24,
                                          output_file='./result/calendar_predict.csv',
                                          return_full=True)

if __name__ == "__main__":
    config=Config()
    run_prediction(config,mode='train')
