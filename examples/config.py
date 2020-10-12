# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# This script is to generate the config for models

import json
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--use_model', type=str, default='seq2seq', help='use model for train, seq2seq, wavenet, transformer')
parser.add_argument('--data_dir', type=str, default='../data/international-airline-passengers.csv', help='dataset directory')
parser.add_argument('--model_dir', type=str, default='../weights/checkpoint', help='saved checkpoint directory')
parser.add_argument('--saved_model_dir', type=str, default='../weights', help='saved pb directory')
parser.add_argument('--log_dir', type=str, default='../data/logs', help='saved pb directory')
parser.add_argument('--input_seq_length', type=int, default=32, help='sequence length for input')
parser.add_argument('--output_seq_length', type=int, default=3, help='sequence length for output')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate for training')

args = parser.parse_args()
params = vars(args)


class Config(object):
    def __init__(self):
        self.params = defaultdict()

    def from_json_file(self, json_file):
        with open(json_file, 'r') as f:
            self.params = json.load(f)

    def to_json_string(self, json_file, params):
        with open(json_file, 'w') as f:
            json.dump(params, f)


if __name__ == '__main__':
    config = Config()
    config.to_json_string('./config.json', params)
    #config.from_json_file('./config.json')
