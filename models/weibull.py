
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

    def predict_by_interval(self,predicted_number):
        x_max=max(self.x)
        #assert x_max< (len(self.x)+5)
        x_predict=[i for i in np.arange(x_max+1,x_max+predicted_number+1)]
        x_predict_weibull=np.log(x_predict).reshape(-1,1)
        y_predict_weibull=self.weight_weibull*x_predict_weibull+self.bias_weibull


    def predict_by_calendar(self):
        pass

    def plot(self):
        fig,ax=plt.subplots()
        ax.plot(self.x_weibull,self.y_weibull,marker='o',linestyle='')
        x_plot = np.arange(np.log(1), np.log(150), np.log(2))
        ax.plot(x_plot,self.weight_weibull*x_plot+self.bias_weibull,'--',linewidth=2)

        ax.set_yticks(list(map(lambda y: np.log(-np.log(1-y)+alpha),np.array([0.01,0.05]+[i/100 for i in range(10,100,20)]))))
        ax.set_yticklabels(np.array([1, 5] + [i for i in range(10, 100, 20)]), fontsize=15)
        ax.set_xticks(list(map(lambda x: np.log(x+alpha),range(0, 60, 10))))
        ax.set_xticklabels(range(0, 60, 10))
        ax.set_xlim([1, np.log(60)])
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Operating months', fontsize=25)
        ax.set_ylabel('Failure probability [%]', fontsize=25)
        #plt.savefig('Test.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
        plt.show()


    def _weibull_density(self,data):
        '''Use cumulative weibull instead of density weibull is that the bin width choose could be zero'''
        loc,scale=stats.weibull_min.fit(data,floc=0)


    def _weibull_density_plot(self,time,failure_rate):
        '''the histogram and curve of density weibull'''
        fig, ax = plt.subplots()
        ax.plot(time, failure_rate)

        ax.set(xlabel='time', ylabel='failure rate interval',title='Failure rate by time')
        ax.grid()
        #fig.savefig("test.png")
        plt.show()