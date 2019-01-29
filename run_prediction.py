import models.weibull
import models.LSTM
from prepare_data import Prepare_Denza,Prepare_Mercedes
import prepare_model_input

class Config:
    pass

def main(config):
    denza_300 = Prepare_Denza(interval_type='monthly',
                          production_file='./raw_data/production_2018_300.xlsx',
                          repair_file='./raw_data/failure_2018_300.xlsx',
                          output_file='Weibull_data_300.csv')

    denza_400 = Prepare_Denza(interval_type='monthly',
                              production_file='./raw_data/production_2018_400.xlsx',
                              repair_file='./raw_data/failure_2018_400.xlsx',
                              output_file='Weibull_data_400.csv')

    input_builder=prepare_model_input.Input_builder('data/Weibull_data_300.csv')
    x1,y1=input_builder.create_weibull_data()

    input_builder2 = prepare_model_input.Input_builder('data/Weibull_data_400.csv')
    x2, y2 = input_builder2.create_weibull_data()

    weibull_model=models.weibull.Weibull_model()
    weight1,bias1=weibull_model.train(x1,y1)

    weibull_model2 = models.weibull.Weibull_model()
    weight2, bias2 = weibull_model2.train(x2, y2)


    weibull_model.plot_two(x1,y1,weight1,bias1,x2,y2,weight2,bias2)

    #weibull_model.predict_by_interval(predicted_interval=12)
    weibull_model.predict_by_calendar(interval_samples=denza_300.data_aggby_interval['Samples'],
                                      calendar_failures=denza_300.failures_aggby_calendar,
                                      predicted_interval=12,
                                      output_file='./output/calendar_predict1.csv')
    weibull_model2.predict_by_calendar(interval_samples=denza_400.data_aggby_interval['Samples'],
                                       calendar_failures=denza_400.failures_aggby_calendar,
                                       predicted_interval=12,
                                       output_file='./output/calendar_predict2.csv')



if __name__=="__main__":
    config=Config()
    main(config)