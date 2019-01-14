import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import optimize

class Weibull(object):
    def __init__(self,x,y):
        alpha=10e-7
        self.x=x
        self.y=y
        self.x_weibull=np.log(x)
        self.y_weibull=np.log(-np.log(1-y)+alpha)

    def train(self):
        self.x_weibull=self.x_weibull.reshape(-1,1)
        self.model=linear_model.LinearRegression()
        self.model.fit(self.x_weibull,self.y_weibull)
        logging.info("Weibull model train finished: {}%".format(self.model.score(self.x_weibull,self.y_weibull)))


    def eval(self):
        pass

    def predict(self):
        pass

    def plot(self):
        fig,ax=plt.subplots()
        ax.plot(self.x,self.y,marker='o',linestyle='')
        weight,bias=optimize.curve_fit(lambda x,A,B: A*x+B,np.log(self.x),np.log(self.y))[0]
        x_plot=np.arange(np.log(1),np.log(150),np.log(2))
        y_plot=weight*x_plot+bias
        plt.plot(np.exp(x_plot),np.exp(y_plot),'--',linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        ax.set_yticks(map(lambda y: -np.log(1-y),np.array([0.01,0.05]+[i/100 for i in range(10,100,20)])))
        ax.set_yticklabels(np.array([1, 5] + [i for i in range(10, 100, 20)]), fontsize=15)
        ax.set_xticks(range(0, 60, 10))
        ax.set_xticklabels(range(0, 60, 10))
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Operating months', fontsize=25)
        ax.set_ylabel('Failure probability [%]', fontsize=25)
        ax.set_xlim([1, 260])
        #plt.savefig('Test.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
        plt.show()


    def _decode_y(self,y):
        pass

    def _weibull_density(self):
        pass

    def _weibull_density_plot(self,time,failure_rate):
        fig, ax = plt.subplots()
        ax.plot(time, failure_rate)
        ax.set(xlabel='time', ylabel='failure rate interval',title='Failure rate by time')
        ax.grid()
        #fig.savefig("test.png")
        plt.show()