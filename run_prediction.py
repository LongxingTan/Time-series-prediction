import models.weibull
import models.LSTM
from prepare_data import Prepare_Denza,Prepare_Mercedes
import prepare_model_input

class Config:
    data_dir='data/Weibull_data.csv'

def main(config):
    denza = Prepare_Denza(interval_type='monthly')
    input_builder=prepare_model_input.Input_builder(config.data_dir)
    x,y=input_builder.create_weibull_data()

    weibull_model=models.weibull.Weibull_model()
    weibull_model.train(x,y)
    weibull_model.plot()

    #weibull_model.predict_by_interval(predicted_interval=12)
    weibull_model.predict_by_calendar(interval_samples=denza.data_aggby_interval['Samples'],
                                      calendar_failures=denza.failures_aggby_calendar,
                                      predicted_interval=12)



if __name__=="__main__":
    config=Config()
    main(config)