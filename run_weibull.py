# Weibull is a common used survival model
# in business, if it's possible to run survival predictions for all possible issues, that's a nice picture
__author__='LongxingTan'

import models.weibull
from prepare_data import Prepare_Denza,Prepare_Mercedes
import prepare_model_input

class Config(object):
    interval_type = 'monthly'
    production_file = './raw_data/samples'
    failure_file = './raw_data/failures'
    model_input_data='./data/Weibull_data.csv'

def main(config):
    mercedes = Prepare_Mercedes(interval_type=config.interval_type,
                                production_file=config.production_file,
                                repair_file=config.failure_file,
                                output_file=config.model_input_data)

    input_builder=prepare_model_input.Input_builder(config.model_input_data)
    x,y=input_builder.create_weibull_input()

    weibull_model=models.weibull.Weibull_model()
    weight,bias=weibull_model.train(x,y)
    weibull_model.plot()
    weibull_model.predict_by_calendar(interval_samples=mercedes.data_aggby_interval['Samples'],
                                      calendar_failures=mercedes.failures_aggby_calendar,
                                      predicted_interval=12,
                                      output_file='./result/calendar_predict.csv')

if __name__=="__main__":
    config=Config()
    main(config)