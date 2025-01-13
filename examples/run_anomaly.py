"""Demo of time series anomaly detection
- https://keras.io/examples/timeseries/timeseries_anomaly_detection/
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tfts import AutoConfig, AutoModel, AutoModelForAnomaly, KerasTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, required=False, help="seed")
    parser.add_argument("--use_model", type=str, default="rnn", help="model for train")
    parser.add_argument("--use_data", type=str, default="ecg", help="dataset: sine or airpassengers")
    parser.add_argument("--train_length", type=int, default=12, help="sequence length for train")
    parser.add_argument("--predict_sequence_length", type=int, default=1, help="sequence length for predict")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--output_dir", type=str, default="./weights", help="saved model weights")
    return parser.parse_args()


def create_subseq(ts, train_length, pred_length):
    sub_seq, next_values = [], []
    for i in range(len(ts) - train_length - pred_length):
        sub_seq.append(ts[i : i + train_length])
        next_values.append(ts[i + train_length : i + train_length + pred_length].T[0])
    return sub_seq, next_values


def build_data(data_name="ecg"):
    if data_name == "ecg":
        df = pd.read_csv("http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt", header=None, delimiter="\t")
        ecg = df.iloc[:, 2].values
        ecg = ecg.reshape(len(ecg), -1)
        print("length of ECG data : ", len(ecg))

        scaler = StandardScaler()
        std_ecg = scaler.fit_transform(ecg)
        std_ecg = std_ecg[:5000]

        sub_seq, next_values = create_subseq(std_ecg, args.train_length, args.predict_sequence_length)
        return np.array(sub_seq), np.array(next_values), std_ecg
    else:
        raise ValueError()


def run_train(args):
    x_test, y_test, sig = build_data("ecg")

    config = AutoConfig.for_model(args.use_model)
    config.train_sequence_length = args.train_length
    model = AutoModelForAnomaly.from_config(config, predict_sequence_length=1)

    trainer = KerasTrainer(model)
    trainer.train((x_test, y_test), (x_test, y_test), epochs=args.epochs)
    # model.save_weights(args.output_dir)
    trainer.save_model(args.output_dir)
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
    # plt.savefig('./anomaly.png')
    plt.show()


def run_inference(args):
    x_test, y_test, sig = build_data("ecg")

    config = AutoConfig.for_model(args.use_model)
    config.train_sequence_length = args.train_length

    model = AutoModelForAnomaly.from_pretrained(weights_dir=args.output_dir)
    det = model.detect(x_test, y_test)
    return sig, det


if __name__ == "__main__":
    args = parse_args()
    run_train(args)

    sig, det = run_inference(args)
    plot(sig, det)
