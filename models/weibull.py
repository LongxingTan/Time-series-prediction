# weibull is a time series prediction model widely used in survival analysis
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
from create_model_input import Input_builder

alpha=10e-7
class Weibull_model(object):
    def __init__(self):
        pass

    def train(self,x,y,train_window):
        self.x = x
        self.y = y
        train_x, train_y = Input_builder().create_weibull_input(x, y, train_windows=train_window)
        self.x_weibull = np.log(train_x)
        self.y_weibull = np.log(-np.log(1 - train_y) + alpha)
        self.x_weibull=self.x_weibull.reshape(-1,1)
        self.model=linear_model.LinearRegression()
        self.model.fit(self.x_weibull,self.y_weibull)
        print("Weibull model train finished with score: {}%".format(100*self.model.score(self.x_weibull,self.y_weibull)))
        self.weight_weibull=self.model.coef_[0]
        self.bias_weibull=self.model.intercept_
        print("Weight: %s, Bias: %s" % (self.weight_weibull,self.bias_weibull))
        return self.weight_weibull,self.bias_weibull

    def predict_by_interval(self,predicted_interval,return_full=True):
        x_max=max(self.x)
        #assert x_max< (len(self.x)+5)
        x_future=[i for i in np.arange(x_max+1,x_max+predicted_interval+1)]
        x_future_weibull=np.log(x_future).reshape(-1,1)
        y_predict_weibull=self.weight_weibull*x_future_weibull+self.bias_weibull
        y_predict=1.0-1.0/(np.exp(np.exp(y_predict_weibull-alpha)))
        y_predict=y_predict.reshape(y_predict.shape[0],)

        if return_full:
            self.x_future = list(self.x) + list(x_future)
            self.y_future = list(self.y) + list(y_predict)
            return self.x_future, self.y_future
        else:
            return list(x_future),list(y_predict)


    def predict_by_calendar(self,interval_samples,calendar_failures,predicted_interval,output_file,return_full=True,include_future_samples=False):
        """
        interval_samples: [sample0,sample1,sample2] cumulative value for each operating time
        calendar_failures: {Date1: failure1,Date2: failure2}
        """
        x_future,y_future=self.predict_by_interval(predicted_interval+1)
        failure_rate_interval =[y_future[i+1]-y_future[i] for i in range(len(y_future)-1)]
        failure_rate_interval=[max(alpha,x) for x in failure_rate_interval]
        logging.info("Use {} data to further predict {} future".format(len(interval_samples),predicted_interval))
        assert len(interval_samples)+predicted_interval==len(failure_rate_interval)

        if return_full:
            pass
        else:
            calendar_failures=pd.DataFrame()

        if include_future_samples:
            #Todo : check if it's better or not
            samples_future=interval_samples+[np.mean(interval_samples)]*predicted_interval

        else:
            for i in range(1,predicted_interval+1):
                samples_failure_rate_interval=failure_rate_interval[i:i+len(interval_samples)]
                failure_interval=sum(interval_samples*samples_failure_rate_interval)
                calendar_failures=calendar_failures.append({"Failures":failure_interval},ignore_index=True)
        calendar_failures.to_csv(output_file,index=False)
        return calendar_failures

    def plot(self):
        plt.style.use('ggplot')
        fig,ax=plt.subplots()
        x, y = np.array(self.x), np.array(self.y)
        x_weibull,y_weibull=np.log(x),np.log(-np.log(1 - y) + alpha)
        ax.plot(x_weibull,y_weibull,marker='o',linestyle='')
        x_plot = np.arange(np.log(1), np.log(len(x)), np.log(2))
        ax.plot(x_plot,self.weight_weibull*x_plot+self.bias_weibull,'--',linewidth=2)

        ax.set_yticks(list(map(lambda y: np.log(-np.log(1 - y) + alpha),np.array([0.01,0.05]+[i/100 for i in range(10,100,20)]))))
        ax.set_yticklabels(np.array([1, 5] + [i for i in range(10, 100, 20)]), fontsize=15)
        ax.set_xticks(list(map(lambda x: np.log(x),[i*10 for i in range(1, 10)])))
        ax.set_xticklabels([i*10 for i in range(1, 10)])
        ax.set_xlim([1, np.log(len(self.x)+10)])
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Operating months', fontsize=25)
        ax.set_ylabel('Failure probability [%]', fontsize=25)
        plt.savefig('./result/weibull.png', format='png', bbox_inches='tight', transparent=False, dpi=600)
        #plt.show()

    def plot_two(self,x1,y1,x2,y2):
        plt.style.use('ggplot')
        fig, ax = plt.subplots()

        x1,y1=np.array(x1),np.array(y1)
        x2,y2=np.array(x2),np.array(y2)

        x1_weibull,x2_weibull=np.log(x1),np.log(x2)
        y1_weibull,y2_weibull=np.log(-np.log(1 - y1) + alpha),np.log(-np.log(1 - y2) + alpha)

        ax.plot(x1_weibull,y1_weibull, marker='o', linestyle='',color='red', markersize=3)
        ax.plot(x2_weibull, y2_weibull, marker='o', linestyle='',color='blue', markersize=2)
        #x_plot = np.arange(np.log(1), np.log(120), np.log(2))
        #ax.plot(x_plot, w1 * x_plot + b1, '--', linewidth=2)
        #ax.plot(x_plot,w2*x_plot+b2,'--',linewidth=2)

        ax.set_yticks(list(map(lambda y: np.log(-np.log(1 - y) + alpha),
                               np.array([0.01, 0.05] + [i / 100 for i in range(10, 100, 20)]))))
        ax.set_yticklabels(np.array([1, 5] + [i for i in range(10, 100, 20)]), fontsize=5)
        ax.set_xticks(list(map(lambda x: np.log(x), [i * 10 for i in range(1, 10)])))
        ax.set_xticklabels([i * 10 for i in range(1, 10)],fontsize=5)
        ax.set_xlim([1, np.log(list(x2)[-1])])
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.set_xlabel('Operating weeks', fontsize=10)
        ax.set_ylabel('Failure probability [%]', fontsize=10)
        plt.savefig('./result/weibull_two.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
        #plt.show()

    def plot_calendar(self,failures_predict,further_true=None):
        true=failures_predict.loc[~failures_predict.iloc[:,0].isnull(),:]
        predict=failures_predict.loc[failures_predict.iloc[:,0].isnull(),:]
        plt.figure(figsize=(14, 5), dpi=100)
        plt.xticks(range(len(true)), true.iloc[:,0], rotation='vertical')
        plt.plot(range(len(true)), true['Failures'], label='true')
        plt.plot([i+len(true) for i in range(len(predict))], predict['Failures'], label='predict')
        if further_true is not None:
            plt.plot([i+len(true) for i in range(len(further_true))],further_true['Failures'],label='true')
        plt.xlabel('Date')
        plt.ylabel('Failures')
        plt.title('Figure 1')
        plt.legend()
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
        fig.savefig("./result/weibull_density.png")


# weibull mixed
# weibull auto regressive
# weibull kernel

class Mixed_weibull(object):
    def __init__(self):
        pass
