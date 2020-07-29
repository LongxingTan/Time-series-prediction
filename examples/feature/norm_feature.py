
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class FeatureNorm(object):
    def __init__(self, type='minmax'):
        self.type = type

    def __call__(self, x, mode='train', model_dir='../models', name='scaler'):
        assert len(x.shape) == 2, "Input rank for FeatureNorm should be 2"
        if self.type == 'standard':
            scaler = StandardScaler()
        elif self.type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported norm type yet: {}".format(self.type))

        if mode=='train':
            scaler.fit(x)
            joblib.dump(scaler, os.path.join(model_dir, name+'.pkl'))
        else:
            scaler = joblib.load(os.path.join(model_dir, name+'.pkl'))
        output = scaler.transform(x)
        try:
            return pd.DataFrame(output, index=x.index, columns=x.columns)
        except:
            return output
