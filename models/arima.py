#https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import logging

class Time_ARIMA():
    def __init__(self,params=None):
        pass

    def train(self,x,grid_search=False):
        # ARIMA order, ARIMA(p,d,q)
        # p is the number of autoregressive terms,
        # d is the number of nonseasonal differences needed for stationarity
        # q is the number of lagged forecast errors in the prediction equation
        if grid_search:
            order, seasonal_order=self.grid_search(x)
        else:
            order,seasonal_order=(3,1,0),(0, 1, 1, 12)
        self.model = sm.tsa.statespace.SARIMAX(x, order=order,seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
        self.model_result =self.model.fit(disp=0)
        logging.info(self.model_result.summary())
        #self.model_result.plot_diagnostics(figsize=(15, 12))
        return self.model_result

    def eval(self,dynamic):
        output = self.model_result.get_prediction(dynamic=dynamic, full_results=False)
        return output.predicted_mean


    def predict(self,test,predict_window):
        pred_uc=self.model_result.get_forecast(steps=predict_window)
        predictions=pred_uc.predicted_mean
        pred_ci = pred_uc.conf_int()
        predictions_low,predictions_high=pred_ci[:, 0],pred_ci[:, 1]
        return predictions,predictions_low,predictions_high


    def plot(self,train,predictions,test=None,predictions_low=None,predictions_high=None):
        plt.plot(range(len(train)),train,label='true',color='blue')
        plt.plot([i+len(train) for i in range(len(predictions))],predictions, color='red',label='predictions')
        if test is not None:
            plt.plot([i+len(train) for i in range(len(test))],test,color='blue',label='true')
        if predictions_high is not None and predictions_low is not None:
            plt.fill_between([i+len(train) for i in range(len(predictions))],predictions_low,predictions_high,color='k', alpha=.25)
        plt.show()

        #self.model_fit.plot_diagnostics(figsize=(20, 14))
        #plt.show()

    def grid_search(self,train_data):
        p=range(0,4)
        d=range(0,4)
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

    def __str__(self):
        return "arima"
