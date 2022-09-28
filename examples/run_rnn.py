
import numpy as np
import tensorflow as tf
from tfts import AutoModel, AutoConfig, KerasTrainer
from dataset import AutoData
from config import parse_args
from utils import set_seed


def build_data():
    return


def build_model():
    return


def run_train(args):
    return


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    run_train(args)

