import os
import logging
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
import xgboost as xgb
import operator
import pandas as pd
import matplotlib.pyplot as plt


class Time_XGB(object):
    def __init__(self,params=None):
        self.params_sk = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'seed': 42,
            'eta': 0.1,
            'max_depth': 10,
        }


    def train(self,train,grid_search=False):
        x,y=train[:,1:],train[:,0]
        if grid_search:
            self.params_sk=self.grid_search(x,y)
        model = XGBRegressor(**self.params_sk)
        self.model_final=model.fit(x,y)


    def eval(self):
        pass

    def predict(self,test,predict_window=None):
        x,y=test[:,1:],test[:,0]
        predictions=self.model_final.predict(x)
        return predictions,None,None

    def plot(self,train,test,predictions,predictions_low,predictions_high):
        train=train[:,0]
        plt.plot(range(len(train)), train, label='true', color='blue')
        plt.plot([i + len(train) for i in range(len(predictions))], predictions, color='red', label='predictions')
        if test is not None:
            test=test[:,0]
            plt.plot([i + len(train) for i in range(len(test))], test, color='blue', label='true')
        if predictions_high is not None and predictions_low is not None:
            plt.fill_between([i + len(train) for i in range(len(predictions))], predictions_low, predictions_high,
                             color='k', alpha=.25)
        plt.show()

    def feature_importance(self,x_train,y_train,x_test,y_test,n_tree,early_stop=True):
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test, y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'validate')]
        xgb_model = xgb.train(self.params_sk, dtrain, n_tree, evals=watchlist,
                              early_stopping_rounds=early_stop, verbose_eval=True)
        importance = xgb_model.get_fscore()
        importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
        self.feature_importance_plot(importance_sorted)


    def restore(self):
        path = os.path.join('./result/checkpoint', 'xgb.joblib.dat')
        self.model=joblib.load(path)

    def save(self):
        path = os.path.join('./result/checkpoint', 'xgb.joblib.dat')
        logging.info('Saving the model to:', path)
        joblib.dump(self.model, path)

    def feature_importance_plot(self,importance_sorted):
        df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        plt.figure()
        df.plot(kind='barh', x='feature', y='fscore',legend=False, figsize=(12, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.tight_layout()
        plt.show()

    def grid_search(self,x,y):
        model = XGBRegressor(**self.params_sk)
        params_grid={
            'n_estimator':st.randint(100,500),
            'max_depth':st.randint(6,20)
        }
        search_sk=RandomizedSearchCV(model,params_grid,cv=5,random_state=1,n_iter=20)
        search_sk.fit(x,y)
        # best parameters
        print("best parameters:", search_sk.best_params_)
        print("best score:", search_sk.best_score_)
        # with new parameters
        params_new = {**self.params_sk, **search_sk.best_params_}
        return params_new

    def random_search(self):
        pass

    def __str__(self):
        return "xgb"
