#https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools

class Time_ARIMA():
    def __init__(self,params=None):
        pass

    def train(self,x):
        # ARIMA order, ARIMA(p,d,q)
        # p is the number of autoregressive terms,
        # d is the number of nonseasonal differences needed for stationarity
        # q is the number of lagged forecast errors in the prediction equation
        self.model = ARIMA(x, order=(3,1,0))
        self.model_fit =self.model.fit(disp=0)
        #print(self.model_fit.summary())

    def eval(self):
        pass

    def predict_point(self,train,test,predict_window):
        history = [x for x in train]
        predictions=[]
        for t in range(predict_window):
            self.model = ARIMA(history, order=(5, 1, 0))
            model_fit=self.model.fit(disp=0)
            output=model_fit.forecast()
            yhat=output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions

    def predict(self,train,predict_window):
        self.model.predict(predict_window)

    def plot(self,train,predictions,test=None):
        plt.plot(range(len(train)),train,label='true',color='blue')
        plt.plot([i+len(train) for i in range(len(predictions))],predictions, color='red',label='predictions')
        if test is not None:
            plt.plot([i+len(train) for i in range(len(test))],test,color='blue',label='true')
        plt.show()

        #self.model_fit.plot_diagnostics(figsize=(20, 14))
        #plt.show()

    def grid_search(self,train_data):
        p=range(0,3)
        d=range(0,3)
        q=range(0,5)
        pdq=list(itertools.product(p,d,q))
        seasonal_pdq=[(x[0],x[1],x[2],12) for x in pdq]

        AIC=[]
        SARIMAX_model = []
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train_data,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()

                    print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                    AIC.append(results.aic)
                    SARIMAX_model.append([param, param_seasonal])
                except:
                    continue

        print('Minimum AIC {}'.format(min(AIC)))
        order=SARIMAX_model[AIC.index(min(AIC))][0]
        seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1]
        return order,seasonal_order
