
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats

alpha=10e-7
class Weibull_model(object):
    def __init__(self):
        pass

    def train(self,x,y):
        self.x = x
        self.y = y
        self.x_weibull = np.log(x)
        self.y_weibull = np.log(-np.log(1 - y) + alpha)
        self.x_weibull=self.x_weibull.reshape(-1,1)
        self.model=linear_model.LinearRegression()
        self.model.fit(self.x_weibull,self.y_weibull)
        print("Weibull model train finished: {}%".format(100*self.model.score(self.x_weibull,self.y_weibull)))
        self.weight_weibull=self.model.coef_[0]
        self.bias_weibull=self.model.intercept_

    def eval(self,x_test,y_test):
        x_test_weibull=np.log(x_test).reshape(-1,1)
        y_test_weibull=np.log(-np.log(1-y_test)+alpha)
        y_test_weibull_hat=self.weight_weibull*x_test_weibull+self.bias_weibull

    def predict_by_interval(self,predicted_interval):
        x_max=max(self.x)
        #assert x_max< (len(self.x)+5)
        x_future=[i for i in np.arange(x_max+1,x_max+predicted_interval+1)]
        x_future_weibull=np.log(x_future).reshape(-1,1)
        y_predict_weibull=self.weight_weibull*x_future_weibull+self.bias_weibull
        y_predict=1-np.exp(-np.exp(y_predict_weibull-alpha))

        self.x_future=list(self.x)+list(x_future)
        self.y_future=list(self.y)+list(y_predict)
        return self.x_future,self.y_future


    def predict_by_calendar(self,interval_samples,calendar_failures,predicted_interval,include_future_samples=False):
        x_future,y_future=self.predict_by_interval(predicted_interval+1)
        failure_rate_interval =[y_future[i+1]-y_future[i] for i in range(len(y_future)-1)]

        assert len(interval_samples)+predicted_interval==len(failure_rate_interval)

        if include_future_samples:
            #to be updated
            samples_future=interval_samples+[np.mean(interval_samples)]*predicted_interval

        else:
            for i in range(1,predicted_interval+1):
                samples_failure_rate_interval=failure_rate_interval[i:i+len(interval_samples)]
                failure_interval=sum(interval_samples*samples_failure_rate_interval)
                calendar_failures=calendar_failures.append({"Failures":failure_interval},ignore_index=True)
        calendar_failures.to_csv('./output/calendar_predict.csv',index=False)


    def plot(self):
        fig,ax=plt.subplots()
        ax.plot(self.x_weibull,self.y_weibull,marker='o',linestyle='')
        x_plot = np.arange(np.log(1), np.log(120), np.log(2))
        ax.plot(x_plot,self.weight_weibull*x_plot+self.bias_weibull,'--',linewidth=2)

        ax.set_yticks(list(map(lambda y: np.log(-np.log(1 - y) + alpha),np.array([0.01,0.05]+[i/100 for i in range(10,100,20)]))))
        ax.set_yticklabels(np.array([1, 5] + [i for i in range(10, 100, 20)]), fontsize=15)
        ax.set_xticks(list(map(lambda x: np.log(x),[i*10 for i in range(1, 10)])))
        ax.set_xticklabels([i*10 for i in range(1, 10)])
        ax.set_xlim([1, np.log(220)])
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Operating months', fontsize=25)
        ax.set_ylabel('Failure probability [%]', fontsize=25)
        plt.savefig('./output/weibull.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
        #plt.show()


    def _weibull_density(self,data):
        '''Use cumulative weibull instead of density weibull is that the bin width choose could be zero'''
        loc,scale=stats.weibull_min.fit(data,floc=0)


    def _weibull_density_plot(self,time,failure_rate):
        '''the histogram and curve of density weibull'''
        fig, ax = plt.subplots()
        ax.plot(time, failure_rate)
        ax.set(xlabel='time', ylabel='failure rate interval',title='Failure rate by time')
        ax.grid()
        fig.savefig("./output/weibull_density.png")
