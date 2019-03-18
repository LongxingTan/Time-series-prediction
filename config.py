import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--model_dir',type=str, default='./result/checkpoint',help='Tensorflow checkpoint directory')
parser.add_argument('--data_dir',type=str, default='./data/example.train',help='train file')
parser.add_argument('--input_seq_length',type=int,default=5,help='sequence length for input')
parser.add_argument('--output_seq_length',type=int,default=5,help='sequence length for output')
parser.add_argument('--lstm_hidden_size',type=int, default=100,help='lstm hidden size')
parser.add_argument('--num_stacked_layers',type=int, default=2,help='lstm hidden size')
parser.add_argument('--dropout_keep',type=float, default=0.95,help='dropout keep prob')
parser.add_argument('--do_train',type=bool, default=True,help='if train the model or not')
parser.add_argument('--do_eval',type=bool, default=True,help='if evaluate the model or not')
parser.add_argument('--do_predict',type=bool, default=True,help='if predict the model or not')
parser.add_argument('--output_dir',type=str, default='./result',help='output dir')

args = parser.parse_args()
params = vars(args)


class Config(object):
    def __init__(self):
        self.params=defaultdict()

    def from_json_file(self,json_file):
        with open(json_file, 'r') as f:
            self.params = json.load(f)

    def to_json_string(self,json_file,params):
        with open(json_file, 'w') as f:
            json.dump(params, f)


if __name__=='__main__':
    config = Config()
    config.to_json_string('./config.json',params)
    #config.from_json_file('./config.json')