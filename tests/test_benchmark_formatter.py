import builtins
import csv
import tempfile
import unittest

from benchmark.formatter import BenchmarkResults
from benchmark.registry import DatasetRegistry


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

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            try:
                builtins.__import__ = import_without_pandas
                results.to_csv(tmp.name)
            finally:
                builtins.__import__ = original_import

            tmp.seek(0)
            rows = list(csv.DictReader(line.decode("utf-8") for line in tmp.readlines()))

        self.assertEqual(rows[0]["dataset"], "synthetic")
        self.assertEqual(rows[0]["model"], "rnn")
        self.assertEqual(rows[0]["metrics"], "{'mae': 0.1}")


class DatasetRegistryTest(unittest.TestCase):
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
