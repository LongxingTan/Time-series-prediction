"""End-to-end time-series anomaly detection with TFTS.

- https://keras.io/examples/timeseries/timeseries_anomaly_detection/
"""

import argparse
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tfts import AutoConfig, AutoModel, AutoModelForAnomaly, KerasTrainer, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, required=False, help="seed")
    parser.add_argument("--use_model", type=str, default="tcn", help="model for train")
    parser.add_argument("--train_length", type=int, default=12, help="sequence length for train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/anomaly_model",
        help="Directory in which to save the trained model",
    )
    return parser.parse_args()


def _output_dir(args):
    """Return a stable artifact directory for one example run."""
    model_dir = getattr(args, "output_dir", None)
    if not model_dir:
        model_dir = tempfile.mkdtemp(prefix="tfts_anomaly_")
        args.output_dir = model_dir
    return model_dir


def create_subsequences(time_series, train_length):
    """Create overlapping input windows to feed the reconstruction model."""
    subsequences = []
    for i in range(len(time_series) - train_length + 1):
        subsequences.append(time_series[i : i + train_length])
    return np.array(subsequences)


def load_and_preprocess_data(args):
    """Load ECG data, scale it, and prepare input windows."""
    url = "http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt"
    df = pd.read_csv(url, header=None, delimiter="\t")
    ecg_data = df.iloc[:, 2].values.reshape(-1, 1)

    print(f"Loaded ECG data of length: {len(ecg_data)}")

    # Standardize the ECG data
    scaler = StandardScaler()
    scaled_ecg = scaler.fit_transform(ecg_data)

    # Create input windows: the anomaly model learns to reconstruct each window.
    windows = create_subsequences(scaled_ecg, args.train_length)
    return windows, scaled_ecg


def build_model(args):
    """Build the reconstruction-based anomaly detection model."""
    config = AutoConfig.for_model(args.use_model)
    config.train_sequence_length = args.train_length
    model = AutoModelForAnomaly.from_config(config)
    return model


def train_model(args, model, train_windows):
    """Train the reconstruction model."""
    set_seed(args.seed)
    trainer = KerasTrainer(model)
    trainer.train(
        train_windows,
        train_windows,
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    # Trainer.save_model stores the TFTS config + weights, so the model can be
    # reconstructed for inference or fine-tuning without Keras custom_objects.
    model_dir = _output_dir(args)
    trainer.save_model(model_dir)
    print(f"Model trained and saved to {model_dir}")


def load_model_for_inference(model_dir, sample_batch):
    """Restore a saved detector; calibration remains an inference step."""
    return AutoModel.from_pretrained(model_dir, sample_batch=sample_batch)


def perform_inference(model, fit_windows, test_windows):
    """Calibrate the threshold on normal data, then score the test windows."""
    model.calibrate(fit_windows)
    output = model.detect(test_windows)
    anomaly_scores = np.asarray(output.scores.numpy()).squeeze()
    anomaly_count = int(tf.reduce_sum(output.labels).numpy())
    print(f"Detected {anomaly_count} anomalous points at threshold {float(output.threshold.numpy()):.4f}")
    return anomaly_scores, test_windows


def plot_results(test_windows, anomaly_scores):
    """Plot the input windows and the anomaly scores."""
    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))

    # Plot a slice of the test signal (first channel of the first windows).
    signal = test_windows[..., 0].flatten()
    axes[0].plot(signal, color="b", label="Test ECG windows")
    axes[0].set_title("ECG Data")
    axes[0].legend()

    axes[1].plot(anomaly_scores, color="r", label="Reconstruction Error")
    axes[1].set_title("Anomaly Detection Scores")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    """Main function to orchestrate training, inference, and plotting."""
    args = parse_args()
    model_dir = _output_dir(args)
    windows, _ = load_and_preprocess_data(args)

    # Split into a fitting split (train + calibrate) and a detection split.
    split = int(len(windows) * 0.8)
    fit_windows, test_windows = windows[:split], windows[split:]

    model = build_model(args)
    train_model(args, model, fit_windows)

    # Restore the saved artifact and run inference on the restored detector.
    restored_model = load_model_for_inference(model_dir, fit_windows[:1])
    anomaly_scores, test_windows = perform_inference(restored_model, fit_windows, test_windows)
    print(f"Loaded model from {model_dir}, inference shape: {anomaly_scores.shape}")
    plot_results(test_windows, anomaly_scores)


if __name__ == "__main__":
    main()
