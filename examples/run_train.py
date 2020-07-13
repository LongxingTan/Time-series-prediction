# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01


import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

from data.load_data import DataLoader
from deepts.model import Model
from config import params


def main():
    data_loader=DataLoader()
    train_dataset=data_loader(params,data_dir=params['data_dir'], batch_size=params['batch_size'],training=True, sample=0.8)
    valid_dataset=data_loader(params,data_dir=params['data_dir'],batch_size=params['batch_size'], training=True, sample=0.2)

    # use_model: seq2seq, wavenet, transformer
    model=Model(params=params, use_model=params['use_model'], use_loss='mse',use_optimizer='adam',custom_model_params={})
    # mode: eager or fit
    model.train(train_dataset,n_epochs=params['n_epochs'],mode='eager',export_model=True)
    model.eval(valid_dataset)


if __name__=='__main__':
    main()
