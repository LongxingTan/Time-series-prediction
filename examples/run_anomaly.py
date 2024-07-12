"""Demo of time series anomaly detection"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, AutoModelForAnomaly, KerasTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, required=False, help="seed")
    parser.add_argument("--use_model", type=str, default="rnn", help="model for train")
    parser.add_argument("--use_data", type=str, default="sine", help="dataset: sine or airpassengers")
    parser.add_argument("--train_length", type=int, default=24, help="sequence length for train")
    parser.add_argument("--predict_length", type=int, default=12, help="sequence length for predict")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")

    return parser.parse_args()


def create_subseq(ts, look_back, pred_length):
    sub_seq, next_values = [], []
    for i in range(len(ts) - look_back - pred_length):
        sub_seq.append(ts[i : i + look_back])
        next_values.append(ts[i + look_back : i + look_back + pred_length].T[0])
    return sub_seq, next_values


def build_data(data_name="ecg"):
    if data_name == "ecg":
        df = pd.read_csv("http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt", header=None, delimiter="\t")
        ecg = df.iloc[:, 2].values
        ecg = ecg.reshape(len(ecg), -1)
        print("length of ECG data : ", len(ecg))

        # standardize
        scaler = StandardScaler()
        std_ecg = scaler.fit_transform(ecg)
        std_ecg = std_ecg[:5000]

        look_back = 10
        pred_length = 3

        sub_seq, next_values = create_subseq(std_ecg, look_back, pred_length)
        return np.array(sub_seq), np.array(next_values), std_ecg
    else:
        raise ValueError()


def run_train(args):
    return


def plot(sig, det):
    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))
    axes[0].plot(sig, color="b", label="original data")
    x = np.arange(4200, 4400)
    y1 = [-3] * len(x)
    y2 = [3] * len(x)
    axes[0].fill_between(x, y1, y2, facecolor="g", alpha=0.3)

    axes[1].plot(det, color="r", label="Mahalanobis Distance")
    axes[1].set_ylim(0, 1000)
    y1 = [0] * len(x)
    y2 = [1000] * len(x)
    axes[1].fill_between(x, y1, y2, facecolor="g", alpha=0.3)
    # plt.savefig('./p.png')
    plt.show()


def run_inference(args):
    model = AutoModelForAnomaly.from_pretrained(args.output_dir)
    model.detect()


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
