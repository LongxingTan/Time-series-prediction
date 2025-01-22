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
    parser.add_argument("--train_length", type=int, default=12, help="sequence length for train")
    parser.add_argument("--predict_sequence_length", type=int, default=1, help="sequence length for predict")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--output_dir", type=str, default="./weights", help="saved model weights")
    return parser.parse_args()


def create_subsequences(time_series, train_length, pred_length):
    """Create subsequences for training and prediction."""
    subsequences, next_values = [], []
    for i in range(len(time_series) - train_length - pred_length):
        subsequences.append(time_series[i : i + train_length])
        next_values.append(time_series[i + train_length : i + train_length + pred_length].T[0])
    return subsequences, next_values


def load_and_preprocess_data(args):
    """Load ECG data, scale it, and prepare subsequences."""
    url = "http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt"
    df = pd.read_csv(url, header=None, delimiter="\t")
    ecg_data = df.iloc[:, 2].values.reshape(-1, 1)

    print(f"Loaded ECG data of length: {len(ecg_data)}")

    # Standardize the ECG data
    scaler = StandardScaler()
    scaled_ecg = scaler.fit_transform(ecg_data)

    # Create subsequences for training and prediction
    subsequences, next_values = create_subsequences(scaled_ecg, args.train_length, args.predict_sequence_length)
    return np.array(subsequences), np.array(next_values), scaled_ecg


def train_model(args):
    """Train the model using the specified arguments."""
    x_train, y_train, _ = load_and_preprocess_data(args)

    config = AutoConfig.for_model(args.use_model)
    config.train_sequence_length = args.train_length
    model = AutoModelForAnomaly.from_config(config, predict_sequence_length=args.predict_sequence_length)

    trainer = KerasTrainer(model)
    trainer.train((x_train, y_train), (x_train, y_train), epochs=args.epochs)
    trainer.save_model(args.output_dir)
    print(f"Model trained and saved to {args.output_dir}")


def perform_inference(args):
    """Perform inference using the trained model."""
    x_test, y_test, _ = load_and_preprocess_data(args)

    print("Starting inference...")
    config = AutoConfig.for_model(args.use_model)
    config.train_sequence_length = args.train_length
    model = AutoModelForAnomaly.from_pretrained(weights_dir=args.output_dir)

    anomaly_scores = model.detect(x_test, y_test)
    return _, anomaly_scores


def plot_results(signal, anomaly_scores):
    """Plot the original signal and detected anomalies."""
    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))

    axes[0].plot(signal, color="b", label="Original Data")
    x_range = np.arange(4200, 4400)
    axes[0].fill_between(x_range, -3, 3, facecolor="g", alpha=0.3)
    axes[0].set_title("ECG Data with Anomalies")
    axes[0].legend()

    axes[1].plot(anomaly_scores, color="r", label="Mahalanobis Distance")
    axes[1].set_ylim(0, 1000)
    axes[1].fill_between(x_range, 0, 1000, facecolor="g", alpha=0.3)
    axes[1].set_title("Anomaly Detection Scores")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    """Main function to orchestrate training, inference, and plotting."""
    args = parse_args()
    train_model(args)

    # Run inference
    signal, anomaly_scores = perform_inference(args)
    plot_results(signal, anomaly_scores)


if __name__ == "__main__":
    main()
