#https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot


class Time_ARIMA():
    def __init__(self,config):
        self.config=config

    def train(self,x):
        # ARIMA order, ARIMA(p,d,q)
        # p is the number of autoregressive terms,
        # d is the number of nonseasonal differences needed for stationarity
        # q is the number of lagged forecast errors in the prediction equation
        self.model = ARIMA(x,order=(5,1,0))
        model_fit =self.model.fit(disp=0)
        print(model_fit.summary())

    def eval(self):
        pass

    def predict(self,train,test):
        history = [x for x in train]
        predictions=[]
        for t in range(len(test)):
            self.model = ARIMA(history, order=(5, 1, 0))
            model_fit=self.model.fit(disp=0)
            output=model_fit.forecast()
            yhat=output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))

    def plot(self,test,predictions):
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()