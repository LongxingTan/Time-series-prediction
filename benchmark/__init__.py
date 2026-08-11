"""TFTS Benchmark System.

A flexible benchmarking framework for evaluating TFTS modelsacross multiple
datasets with multiple metrics and multiple runs. Designed for reproducibility
and paper-quality results.

Example::

    from benchmark import BenchmarkRunner, BenchmarkConfig
    from benchmark.datasets import GrocerysalesDataset, RecruitRestaurantDataset

    config = BenchmarkConfig(
        models=["rnn", "transformer", "dlinear"],
        datasets=["grocery_sales", "recruit_restaurant"],
        metrics=["mae", "rmse", "mape"],
        runs=5,
        output_dir="benchmark_results",
    )

    runner = BenchmarkRunner(config)
    results = runner.run()
    results.to_latex("benchmark_results.tex")
    results.to_csv("benchmark_results.csv")
"""

from benchmark.base import BenchmarkConfig, Dataset
from benchmark.formatter import BenchmarkResults
from benchmark.registry import DatasetRegistry, ModelRegistry
from benchmark.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResults",
    "Dataset",
    "DatasetRegistry",
    "ModelRegistry",
]
