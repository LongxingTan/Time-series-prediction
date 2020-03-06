
import numpy as np
import pandas as pd
from deepts.models.seq2seq import Seq2seq
from deepts.models.tcn import TCN
from deepts.models.transformer import Transformer
from deepts.models.gan import GAN


class Model(object):
    def __init__(self,use_model,loss,optimizer):
        if use_model=='seq2seq':
            Model=Seq2seq()
        elif use_model=='tcn':
            Model=TCN()
        elif use_model=='transformer':
            Model=Transformer()
        elif use_model=='gan':
            Model=GAN()
        else:
            raise ValueError("unsupported use_model of {}".format(use_model))

        self.model=Model(inputs_shape=[10,1],training=True)
        self.model.compile(loss=loss,optimizer=optimizer)

    def fit(self,x,y,epochs):
        if isinstance(x,pd.DataFrame):
            x=x.values
        if len(x.shape)==2:
            x=x[...,np.newaxis]
        if isinstance(y,pd.DataFrame):
            y=y.values
        if len(y.shape)==2:
            y=y[...,np.newaxis]

        self.model.fit([x,y],y,epochs=epochs)

    def predict(self):
        pass
