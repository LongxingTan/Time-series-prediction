import os
import logging
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


class Time_XGB(object):
    def __init__(self,params=None):
        pass

    def build(self):
        params_sk={
            'booster': 'gbtree',
            'objective':'reg:linear',
            'subsample':0.8,
            'colsample_bytree':0.85,
            'seed':42
        }
        self.model=XGBRegressor(**params_sk)

    def train(self,x_train,y_train):
        self.build()
        self.model.fit(x_train,y_train)
        logging.info(self.model.get_fscore)


    def eval(self):
        pass

    def predict(self,train,predict_window):
        self.model.predict(train)

    def plot(self):
        pass

    def restore(self):
        path = os.path.join('./result/checkpoint', 'xgb.joblib.dat')
        self.model=joblib.load(path)

    def save(self):
        path = os.path.join('./result/checkpoint', 'xgb.joblib.dat')
        logging.info('Saving the model to:', path)
        joblib.dump(self.model, path)

    def feature_importance_plot(self):
        importance=self.model.get_fscore()

    def grid_search(self,x,y):
        params_grid={
            'n_estimator':st.randint(100,500),
            'max_depth':st.randint(6,20)
        }
        search_sk=RandomizedSearchCV(self.model,params_grid,cv=5,random_state=1,n_iter=20)
        search_sk.fit(x,y)

    def random_search(self):
        pass
