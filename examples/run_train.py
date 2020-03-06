# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01


from prepare_data import PassengerData
from deepts.models.model import Model

params={}
x,y=PassengerData(params).get_examples(data_dir='./international-airline-passengers.csv')
print(x.values.shape,y.shape)

model=Model(use_model='tcn',loss='mse',optimizer='adam')
model.fit(x,y,epochs=10)