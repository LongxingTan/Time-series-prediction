import models.weibull
import pandas as pd

class Config:
    data_dir='data/Weibull_data.csv'

def main(config):
    examples=pd.read_csv(config.data_dir)
    x,y=examples['Interval'].values,examples['Failure_rate_cum'].values
    weibull_model=models.weibull.Weibull(x,y)
    weibull_model.train()


if __name__=="__main__":
    config=Config()
    main(config)