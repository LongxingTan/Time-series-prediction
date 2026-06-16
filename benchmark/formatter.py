"""Result formatting for the TFTS benchmark system.

Supports console tables, CSV, JSON, and LaTeX output for papers."""

import csv
from dataclasses import asdict
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _format_value(val: Any) -> str:
    """Pretty formatter for values."""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _avg(values: List[float]) -> float:
    """Return mean, ignoring NaNs."""
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if len(arr) > 0 else float("nan")


def _std(values: List[float]) -> float:
    """Return std dev, ignoring NaNs."""
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    return float(np.std(arr)) if len(arr) > 0 else float("nan")


class BenchmarkResults:
    """Container for benchmark results with export helpers.

    Attributes:
        results: Raw list of per-run result dictionaries.
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pivot(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """Pivot raw results into {dataset: {model: {metric: [values]}}}."""
        out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for row in self.results:
            ds_name = row.get("dataset", "unknown")
            model_name = row.get("model", "unknown")
            metrics = row.get("metrics", {})
            if ds_name not in out:
                out[ds_name] = {}
            if model_name not in out[ds_name]:
                out[ds_name][model_name] = {}
            for metric, value in metrics.items():
                if metric not in out[ds_name][model_name]:
                    out[ds_name][model_name][metric] = []
                try:
                    out[ds_name][model_name][metric].append(float(value))
                except (TypeError, ValueError):
                    pass
        return out

    # ------------------------------------------------------------------
    # Public export API
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Return a pandas DataFrame with averaged results (mean std columns)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for ds_name, models in self._pivot().items():
            for model_name, metrics in models.items():
                flat: Dict[str, Any] = {"dataset": ds_name, "model": model_name}
                for metric_name, values in metrics.items():
                    flat[f"{metric_name}_mean"] = _avg(values)
                    flat[f"{metric_name}_std"] = _std(values)
                rows.append(flat)
        return pd.DataFrame(rows)

    def to_csv(self, path: str) -> None:
        """Export results to a CSV file."""
        try:
            import pandas as pd

            self.to_dataframe().to_csv(path, index=False)
            logger.info("Results saved to %s", path)
        except ImportError:
            # Fallback with csv module
            _dicts = [asdict(r) for r in self.results]
            if not _dicts:
                return
            keys = _dicts[0].keys()
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(_dicts)
            logger.info("Results saved to %s", path)

    def to_json(self, path: str) -> None:
        """Export raw results to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        logger.info("Results saved to %s", path)

    def to_latex(self, path: str, metric: Optional[str] = None) -> None:
        """Export averaged results to a LaTeX table suitable for papers.

        Args:
            path: Output .tex file path.
            metric: If given, only include this metric. Otherwise include all.
        """
        lines: List[str] = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Benchmark Results}",
            r"\label{tab:benchmark}",
        ]

        pivot = self._pivot()
        first_dataset = next(iter(pivot)) if pivot else None
        if first_dataset is None:
            lines.extend([r"\begin{tabular}{c}", "No results", r"\end{tabular}", r"\end{table}"])
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            return

        models = sorted(next(iter(pivot.values())).keys())
        metrics = sorted(next(iter(next(iter(pivot.values())).values())).keys())
        target_metrics = [metric] if metric else metrics

        # Build header
        header = "Dataset & Model & " + " & ".join(target_metrics) + r" \\\\"
        lines.append(r"\begin{tabular}{ll" + "r" * len(target_metrics) + "}")
        lines.append(r"\toprule")
        lines.append(header)
        lines.append(r"\midrule")

        for ds_name, model_data in pivot.items():
            for model_name in models:
                vals = model_data.get(model_name, {})
                row_vals = []
                for m in target_metrics:
                    values = vals.get(m, [])
                    if values:
                        mean = _avg(values)
                        std = _std(values)
                        row_vals.append(f"{mean:.4f} $\\pm$ {std:.4f}")
                    else:
                        row_vals.append("-")
                lines.append(f"{ds_name} & {model_name} & " + " & ".join(row_vals) + r" \\\\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        logger.info("LaTeX table saved to %s", path)

    def print_table(self) -> None:
        """Print results to the console as a formatted table."""
        pivot = self._pivot()
        if not pivot:
            print("No results to display.")
            return

        for ds_name, models in pivot.items():
            print(f"\n{'=' * 60}")
            print(f"Dataset: {ds_name}")
            print("=" * 60)
            for model_name, metrics in models.items():
                print(f"\n  Model: {model_name}")
                for metric_name, values in metrics.items():
                    if values:
                        mean = _avg(values)
                        std = _std(values)
                        print(f"    {metric_name:10s}: {mean:10.4f}  ± {std:8.4f}  (n={len(values)})")
                    else:
                        print(f"    {metric_name:10s}: N/A")
