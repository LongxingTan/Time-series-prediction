#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
"""Kaggle dataset example"""

from examples.dataset.read_web_traffic import WebDataReader


class AutoData(object):
    def __init__(self, data_name, train_length, predict_length):
        if data_name == "wtf":
            pass
        elif data_name == "m5":
            pass
        elif data_name == "fgs":
            pass
        else:
            raise ValueError()
