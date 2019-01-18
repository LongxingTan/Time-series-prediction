import models.weibull

import models.LSTM
import prepare_model_input

class Config:
    data_dir='data/Weibull_data.csv'

def main(config):
    input_builder=prepare_model_input.Input_builder(config.data_dir)
    x,y=input_builder.create_weibull_data()

    weibull_model=models.weibull.Weibull_model()
    weibull_model.train(x,y)
    weibull_model.predict_by_interval(12)
    weibull_model.plot()



if __name__=="__main__":
    config=Config()
    main(config)