"""Command-line interface for the TFTS benchmark system.

Usage::

    python -m benchmark.cli --config benchmark/configs/example.yaml

Or with command-line options::

    python -m benchmark.cli \
        --models rnn transformer dlinear \
        --datasets sine air_passengers \
        --metrics mae rmse mape \
        --runs 3 \
        --epochs 50 \
        --output-dir benchmark_results
"""

import argparse
import logging
import sys
from typing import List

from benchmark import BenchmarkConfig, BenchmarkRunner
from benchmark.registry import DatasetRegistry, ModelRegistry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tfts.benchmark")


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tfts-benchmark",
        description="TFTS Benchmark: Run multiple models on multiple datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file. Explicit CLI options override YAML values.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to benchmark (default: all). Use 'all' for every registered model.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to benchmark (default: all). Use 'all' for every registered dataset.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["mae", "mse", "rmse", "mape", "smape", "r2", "mape_pct"],
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of runs per model-dataset pair (for statistical significance).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train-length",
        type=int,
        default=None,
        help="Lookback window. If not set, each dataset uses its default.",
    )
    parser.add_argument(
        "--predict-sequence-length",
        type=int,
        default=None,
        help="Forecast horizon. If not set, each dataset uses its default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Verbosity level (0=silent, 1=progress, 2=detailed).",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also generate a LaTeX table.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
    )
    return parser


def main(argv: List[str] = None) -> int:
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args(argv)

    model_registry = ModelRegistry()
    dataset_registry = DatasetRegistry()
    # Load default datasets
    from benchmark.datasets import (
        AirPassengersDataset,
        CMIDetectSleepStatesDataset,
        ForecastingStickerSalesDataset,
        GrocerysalesDataset,
        M4Dataset,
        NpzDataset,
        RecruitRestaurantDataset,
        SineDataset,
    )

    dataset_registry.register("sine", SineDataset)
    dataset_registry.register("air_passengers", AirPassengersDataset)
    dataset_registry.register("grocery_sales", GrocerysalesDataset)
    dataset_registry.register("m4", M4Dataset)
    dataset_registry.register("npz", NpzDataset)
    dataset_registry.register("recruit_restaurant", RecruitRestaurantDataset)
    dataset_registry.register("forecasting_sticker_sales", ForecastingStickerSalesDataset)
    dataset_registry.register("CMI_detect_sleep_states", CMIDetectSleepStatesDataset)

    if args.list_models:
        print("Available models:")
        for name in model_registry.available_models:
            print("  " + f"- {name}")
        return 0

    if args.list_datasets:
        print("Available datasets:")
        for name in dataset_registry.list_datasets():
            print("  " + f"- {name}")
        return 0

    config = BenchmarkConfig.from_yaml(args.config) if args.config else BenchmarkConfig()
    cli_overrides = {
        "models": args.models,
        "datasets": args.datasets,
        "metrics": args.metrics,
        "runs": args.runs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_length": args.train_length,
        "predict_sequence_length": args.predict_sequence_length,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "verbose": args.verbose,
    }
    for name, value in cli_overrides.items():
        if value is not None:
            setattr(config, name, value)
    config.__post_init__()

    runner = BenchmarkRunner(config, dataset_registry, model_registry)
    results = runner.run()

    # Print to console
    results.print_table()

    if args.latex:
        import os

        latex_path = os.path.join(config.output_dir, "results.tex")
        results.to_latex(latex_path)
        print(f"\nLaTeX table saved to: {latex_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
