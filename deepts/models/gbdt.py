# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-09


class GBDTRegressor(object):
    # This model can predict multiple steps time series prediction by GBDT model, and the main 3 mode can be all included
    def __init__(self, use_model, each_model_per_prediction, extend_prediction_to_train, extend_weights=None):
        pass

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
              categorical_features=None, fit_params={}):
        pass

    def predict(self):
        pass

    def get_feature_importance(self):
        pass

    def _get_null_importance(self):
        pass

    def _get_permutation_importance(self):
        pass

    def _get_lofo_importance(self):
        pass


