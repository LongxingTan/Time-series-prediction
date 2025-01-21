"""Demo of time series classification"""

import argparse

import tfts
from tfts import AutoConfig, AutoModel, AutoModelForClassification, KerasTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=315, required=False, help="seed")
    parser.add_argument("--use_model", type=str, default="bert", help="model for train")
    parser.add_argument("--use_data", type=str, default="sine", help="dataset: sine or airpassengers")
    parser.add_argument("--train_length", type=int, default=24, help="sequence length for train")
    parser.add_argument("--predict_sequence_length", type=int, default=12, help="sequence length for predict")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")

    return parser.parse_args()


def run_train(args):
    config = AutoConfig.for_model(args.use_model)
    model = AutoModelForClassification.from_config(config, num_labels=2)

    print(model.summary())
    return


def run_inference(args):
    return


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
