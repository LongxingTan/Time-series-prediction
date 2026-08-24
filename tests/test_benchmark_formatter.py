import builtins
from contextlib import redirect_stdout
import csv
import io
import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.base import BenchmarkConfig
from benchmark.datasets.m4 import M4Dataset
from benchmark.datasets.npz import NpzDataset
from benchmark.formatter import BenchmarkResults
from benchmark.metrics import BenchmarkMetrics
from benchmark.registry import DatasetRegistry, ModelRegistry
from benchmark.runner import BenchmarkRunner


class BenchmarkResultsTest(unittest.TestCase):
    def test_to_csv_fallback_handles_dict_results_without_pandas(self):
        results = BenchmarkResults(
            [
                {"dataset": "synthetic", "model": "rnn", "metrics": {"mae": 0.1}},
                {"dataset": "synthetic", "model": "tcn", "metrics": {"mae": 0.2}},
            ]
        )

        original_import = builtins.__import__

        def import_without_pandas(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas disabled for fallback test")
            return original_import(name, *args, **kwargs)

        # A NamedTemporaryFile remains open on Windows and cannot be reopened
        # by to_csv.  Use a temporary directory so the exporter can own the
        # output file handle on every platform.
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            try:
                builtins.__import__ = import_without_pandas
                results.to_csv(output_path)
            finally:
                builtins.__import__ = original_import

            with open(output_path, "rb") as output:
                rows = list(csv.DictReader(line.decode("utf-8") for line in output.readlines()))

        self.assertEqual(rows[0]["dataset"], "synthetic")
        self.assertEqual(rows[0]["model"], "rnn")
        self.assertEqual(rows[0]["metrics"], "{'mae': 0.1}")

    def test_dataframe_and_export_formats(self):
        results = BenchmarkResults(
            [
                {"dataset": "a", "model": "rnn", "metrics": {"mae": 1.0, "rmse": 2.0}},
                {"dataset": "a", "model": "rnn", "metrics": {"mae": 3.0, "rmse": "bad"}},
                {"dataset": "b", "model": "tcn", "metrics": {"mae": 4.0}},
            ]
        )

        frame = results.to_dataframe()
        self.assertEqual(set(frame["dataset"]), {"a", "b"})
        row = frame[(frame["dataset"] == "a") & (frame["model"] == "rnn")].iloc[0]
        self.assertEqual(row["mae_mean"], 2.0)
        self.assertEqual(row["mae_std"], 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/results.csv"
            json_path = f"{tmpdir}/results.json"
            results.to_csv(csv_path)
            results.to_json(json_path)
            with open(csv_path, encoding="utf-8") as fh:
                self.assertIn("mae_mean", fh.readline())
            with open(json_path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)[0]["dataset"], "a")

    def test_dataframe_reports_missing_pandas_dependency(self):
        results = BenchmarkResults([])
        original_import = builtins.__import__

        def import_without_pandas(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas disabled for dataframe test")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = import_without_pandas
            with self.assertRaisesRegex(ImportError, "pandas is required"):
                results.to_dataframe()
        finally:
            builtins.__import__ = original_import

    def test_pivot_and_latex_cover_missing_and_invalid_values(self):
        results = BenchmarkResults(
            [
                {"metrics": {"mae": 1.0, "invalid": "not-a-number"}},
                {"dataset": "second", "model": "only", "metrics": {"mae": 2.0}},
            ]
        )
        pivot = results._pivot()
        self.assertEqual(pivot["unknown"]["unknown"]["mae"], [1.0])
        self.assertEqual(pivot["unknown"]["unknown"]["invalid"], [])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.tex")
            results.to_latex(output_path, metric="rmse")
            with open(output_path, "rb") as output:
                latex = output.read().decode("utf-8")
        self.assertIn("rmse", latex)
        self.assertIn("-", latex)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.tex")
            BenchmarkResults([{"dataset": "first", "model": "only", "metrics": {"mae": 1.0}}]).to_latex(
                output_path, metric="mae"
            )
            with open(output_path, encoding="utf-8") as output:
                self.assertIn("1.0000", output.read())

    def test_empty_exports_and_console_output(self):
        empty = BenchmarkResults([])
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/empty.csv"
            original_import = builtins.__import__

            def import_without_pandas(name, *args, **kwargs):
                if name == "pandas":
                    raise ImportError("pandas disabled for empty fallback test")
                return original_import(name, *args, **kwargs)

            try:
                builtins.__import__ = import_without_pandas
                empty.to_csv(csv_path)
            finally:
                builtins.__import__ = original_import
            self.assertFalse(os.path.exists(csv_path))
            latex_path = f"{tmpdir}/empty.tex"
            empty.to_latex(latex_path)
            with open(latex_path, encoding="utf-8") as fh:
                self.assertIn("No results", fh.read())

        output = io.StringIO()
        with redirect_stdout(output):
            empty.print_table()
        self.assertIn("No results to display", output.getvalue())

    def test_print_table_formats_nonempty_results(self):
        results = BenchmarkResults([{"dataset": "synthetic", "model": "rnn", "metrics": {"mae": 0.25}}])
        output = io.StringIO()
        with redirect_stdout(output):
            results.print_table()
        self.assertIn("Dataset: synthetic", output.getvalue())
        self.assertIn("mae", output.getvalue())

        with patch.object(results, "_pivot", return_value={"synthetic": {"rnn": {"mae": []}}}):
            output = io.StringIO()
            with redirect_stdout(output):
                results.print_table()
        self.assertIn("N/A", output.getvalue())


class BenchmarkHelpersTest(unittest.TestCase):
    def test_formatter_helpers_ignore_nan_values(self):
        from benchmark.formatter import _avg, _format_value, _std

        self.assertEqual(_format_value(1.23456), "1.2346")
        self.assertEqual(_format_value("value"), "value")
        self.assertEqual(_avg([1.0, float("nan"), 3.0]), 2.0)
        self.assertEqual(_std([1.0, float("nan"), 3.0]), 1.0)
        self.assertTrue(math.isnan(_avg([])))
        self.assertTrue(math.isnan(_std([float("nan")])))

    def test_benchmark_metrics_cover_standard_and_edge_cases(self):
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([0.0, 1.0, 2.0])
        metrics = BenchmarkMetrics(["mae", "mse", "rmse", "mape", "smape", "r2", "mape_pct"])
        values = metrics.compute(y_true, y_pred)
        self.assertEqual(values["mae"], 1.0)
        self.assertEqual(values["mse"], 5.0 / 3.0)
        self.assertAlmostEqual(values["rmse"], np.sqrt(5.0 / 3.0))
        self.assertIn("mape_pct", values)

        self.assertTrue(math.isnan(BenchmarkMetrics.mape(np.zeros(2), np.ones(2))))
        self.assertTrue(math.isnan(BenchmarkMetrics.smape(np.zeros(2), np.zeros(2))))
        self.assertTrue(math.isnan(BenchmarkMetrics.r2(np.ones(2), np.zeros(2))))
        self.assertEqual(metrics.compute(y_true, y_pred, metrics=["does_not_exist"]), {})
        metrics.mae = lambda *_: (_ for _ in ()).throw(RuntimeError("broken metric"))
        self.assertTrue(math.isnan(metrics.compute(y_true, y_pred, metrics=["mae"])["mae"]))
        with self.assertRaises(ValueError):
            metrics.compute(y_true, np.zeros(2))
        with self.assertRaises(ValueError):
            BenchmarkMetrics(["does_not_exist"])


class BenchmarkRunnerTest(unittest.TestCase):
    def test_registry_resolution_and_runner_validation(self):
        config = BenchmarkConfig(models=["rnn"], datasets=["toy"], output_dir="unused")
        dataset_registry = DatasetRegistry()
        dataset_registry.register("toy", object)
        model_registry = ModelRegistry()
        runner = BenchmarkRunner(config, dataset_registry, model_registry)

        self.assertEqual(runner._resolve_datasets(), ["toy"])
        self.assertEqual(runner._resolve_models(), ["rnn"])
        with self.assertRaises(ValueError):
            BenchmarkRunner(
                BenchmarkConfig(models=["missing"], datasets=["toy"]), dataset_registry, model_registry
            )._resolve_models()
        with self.assertRaises(ValueError):
            BenchmarkRunner(
                BenchmarkConfig(models=["rnn"], datasets=["missing"]), dataset_registry, model_registry
            )._resolve_datasets()

        all_config = BenchmarkConfig(models=["all"], datasets=["all"])
        all_runner = BenchmarkRunner(all_config, dataset_registry, model_registry)
        self.assertEqual(all_runner._resolve_datasets(), ["toy"])
        self.assertIn("rnn", all_runner._resolve_models())

    def test_single_trial_uses_dataset_overrides(self):
        config = BenchmarkConfig(models=["rnn"], datasets=["toy"], epochs=3, batch_size=4, learning_rate=0.1)
        dataset_registry = DatasetRegistry()
        runner = BenchmarkRunner(config, dataset_registry, ModelRegistry())
        dataset = type("ToyDataset", (), {"train_length": 8, "predict_sequence_length": 2})()
        train = (np.zeros((2, 4, 1)), np.zeros((2, 2, 1)))
        valid = (np.zeros((1, 4, 1)), np.zeros((1, 2, 1)))
        dataset.get_train_valid_split = lambda **kwargs: (train, valid)
        history = type("History", (), {"history": {"loss": [np.float32(0.5)]}})()

        with patch("benchmark.runner.AutoConfig") as auto_config, patch(
            "benchmark.runner.AutoModel"
        ) as auto_model, patch("benchmark.runner.Trainer") as trainer_cls:
            auto_config.for_model.return_value = type("Config", (), {"input_shape": None})()
            auto_model.from_config.return_value = object()
            trainer_cls.return_value.train.return_value = history
            trainer_cls.return_value.predict.return_value = valid[1]
            result = runner._run_single_trial(
                dataset,
                "toy",
                "rnn",
                run_idx=0,
                seed=42,
                ds_config={"train_length": 5, "predict_sequence_length": 1, "epochs": 1, "batch_size": 2},
            )

        self.assertEqual(result["train_length"], 5)
        self.assertEqual(result["predict_sequence_length"], 1)
        self.assertEqual(result["history"]["loss"], [0.5])

    def test_run_and_save_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(models=["rnn"], datasets=["toy"], output_dir=tmpdir)
            dataset_registry = DatasetRegistry()
            dataset_registry.register("toy", object)
            runner = BenchmarkRunner(config, dataset_registry, ModelRegistry())
            with patch.object(runner, "_run_experiment") as run_experiment:
                result = runner.run()
            run_experiment.assert_called_once_with("toy", "rnn")
            self.assertEqual(result.results, [])

    def test_experiment_runs_each_seed_and_default_registry_is_available(self):
        config = BenchmarkConfig(models=["rnn"], datasets=["toy"], runs=2, seed=10)
        dataset_registry = DatasetRegistry()

        class ToyDataset:
            train_length = 4
            predict_sequence_length = 1

        dataset_registry.register("toy", ToyDataset)
        runner = BenchmarkRunner(config, dataset_registry, ModelRegistry())
        trial_results = [{"run": 0}, {"run": 1}]
        with patch("benchmark.runner.set_seed") as set_seed, patch.object(
            runner, "_run_single_trial", side_effect=trial_results
        ) as run_trial:
            runner._run_experiment("toy", "rnn")
        self.assertEqual(runner.results, trial_results)
        self.assertEqual(set_seed.call_args_list[0].args, (10,))
        self.assertEqual(set_seed.call_args_list[1].args, (11,))
        self.assertEqual(run_trial.call_count, 2)

        from benchmark.runner import _default_dataset_registry

        default_names = _default_dataset_registry().list_datasets()
        self.assertIn("sine", default_names)
        self.assertIn("grocery_sales", default_names)


class BenchmarkConfigTest(unittest.TestCase):
    def test_validation_and_dataset_overrides(self):
        with self.assertRaises(ValueError):
            BenchmarkConfig(runs=0)
        with self.assertRaises(ValueError):
            BenchmarkConfig(epochs=0)
        config = BenchmarkConfig(
            epochs=3,
            batch_size=4,
            per_dataset_config={"toy": {"epochs": 1, "train_length": 5}},
        )
        self.assertEqual(config.get_dataset_config("toy")["epochs"], 1)
        self.assertEqual(config.get_dataset_config("other")["epochs"], 3)

    def test_loads_yaml_with_dataset_and_model_overrides(self):
        yaml_text = """
benchmark:
  epochs: 7
  metrics: [mae]
  output_dir: custom-results
models:
  rnn:
    hidden_size: 16
  transformer: {}
datasets:
  sine:
    train_length: 12
    n_examples: 50
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "benchmark.yaml")
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write(yaml_text)
            config = BenchmarkConfig.from_yaml(config_path)

        self.assertEqual(config.models, ["rnn", "transformer"])
        self.assertEqual(config.datasets, ["sine"])
        self.assertEqual(config.metrics, ["mae"])
        self.assertEqual(config.get_dataset_config("sine")["train_length"], 12)
        self.assertEqual(config.get_dataset_config("sine")["n_examples"], 50)
        self.assertEqual(config.get_model_config("rnn"), {"hidden_size": 16})

    def test_nested_model_sections(self):
        yaml_text = """
models:
  tft:
    config: {hidden_size: 16}
    training: {loss: multi_quantile}
    prediction: {quantile: 0.5}
datasets: [sine]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "benchmark.yaml")
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write(yaml_text)
            config = BenchmarkConfig.from_yaml(config_path)
        self.assertEqual(config.get_model_config("tft"), {"hidden_size": 16})
        self.assertEqual(config.get_model_training("tft"), {"loss": "multi_quantile"})
        self.assertEqual(config.get_model_prediction("tft"), {"quantile": 0.5})

    def test_yaml_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "benchmark.yaml")
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write("models: rnn\n")
            with self.assertRaisesRegex(ValueError, "models"):
                BenchmarkConfig.from_yaml(config_path)


class RegistryTest(unittest.TestCase):
    def test_base_registry_and_model_registry_operations(self):
        registry = DatasetRegistry()
        registry.register("toy", int)
        self.assertIn("toy", registry)
        self.assertEqual(registry.get("toy"), int)
        self.assertEqual(registry.list_items(), {"toy": int})
        with self.assertLogs("benchmark.registry", level="WARNING"):
            registry.register("toy", str)
        with self.assertRaises(KeyError):
            registry.get("missing")

        models = ModelRegistry()
        models.register("custom", "CustomModel")
        self.assertIn("custom", models)
        self.assertEqual(models.get("custom"), "CustomModel")
        with self.assertRaises(KeyError):
            models.get("missing")


class DatasetRegistryTest(unittest.TestCase):
    def test_m4_dataset_builds_complete_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import pandas as pd

            pd.DataFrame({"M4id": ["W1", "M1"], "SP": ["Weekly", "Monthly"]}).to_csv(
                os.path.join(tmpdir, "M4-info.csv"), index=False
            )
            train_values = np.empty(2, dtype=object)
            train_values[:] = [np.arange(30, dtype=np.float32), np.arange(20, dtype=np.float32)]
            test_values = np.empty(2, dtype=object)
            test_values[:] = [np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32)]
            with open(os.path.join(tmpdir, "training.npz"), "wb") as output:
                np.save(output, train_values, allow_pickle=True)
            with open(os.path.join(tmpdir, "test.npz"), "wb") as output:
                np.save(output, test_values, allow_pickle=True)

            train, valid = M4Dataset().get_train_valid_split(
                data_dir=tmpdir,
                seasonal_pattern="Weekly",
                train_length=8,
                predict_sequence_length=4,
                windows_per_series=2,
                window_seed=1,
            )
        self.assertEqual(train[0].shape, (2, 8, 1))
        self.assertEqual(train[1].shape, (2, 4, 1))
        self.assertEqual(valid[0].shape, (1, 8, 1))
        self.assertEqual(valid[1].shape, (1, 4, 1))

    def test_npz_dataset_supports_array_and_structured_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.npz")
            valid_path = os.path.join(tmpdir, "valid.npz")
            values = np.zeros((3, 4, 1), dtype=np.float32)
            targets = np.ones((3, 2, 1), dtype=np.float32)
            np.savez(train_path, x=values, known=values, y=targets)
            np.savez(valid_path, x=values, known=values, y=targets)

            dataset = NpzDataset()
            train, valid = dataset.get_train_valid_split(
                train_path=train_path,
                valid_path=valid_path,
                input_keys=["x", "known"],
                target_key="y",
            )
        self.assertEqual(set(train[0]), {"x", "known"})
        np.testing.assert_array_equal(valid[1], targets)

    def test_lazy_dataset_registration_returns_instantiable_wrapper(self):
        class ToyDataset:
            def prepare_data(self, **kwargs):
                return "prepared"

            def get_train_valid_split(self, **kwargs):
                return "split"

        registry = DatasetRegistry()
        registry.register_lazy("toy", lambda: ToyDataset())

        dataset_cls = registry.get("toy")
        dataset = dataset_cls()

        self.assertEqual(dataset_cls.name, "toy")
        self.assertEqual(dataset.prepare_data(), "prepared")
        self.assertEqual(dataset.get_train_valid_split(), "split")


if __name__ == "__main__":
    unittest.main()
