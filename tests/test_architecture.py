"""Enforce the ownership and data boundaries established by the restructure."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tfts"


def imports(path):
    module = path.relative_to(ROOT).with_suffix("").parts
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = module[:-1]
                base = parent[: len(parent) - node.level + 1]
                yield ".".join(base + tuple((node.module or "").split(".")))
            else:
                yield node.module or ""


class ArchitectureTest(unittest.TestCase):
    def test_removed_paths_have_no_shims(self):
        for name in (
            "contracts",
            "features",
            "models/registry.py",
            "models/auto_model.py",
            "tasks/task_models.py",
            "data/frame.py",
            "data/processor.py",
            "data/timeseries.py",
            "data/auto_preprocessor.py",
            "data/sequence_utils.py",
            "data/windowing.py",
            "data/window_sampling.py",
            "data/_window_geometry.py",
            "data/materializers.py",
            "data/preprocessing.py",
            "data/get_data.py",
            "data/ar.py",
            "data/examples",
        ):
            with self.subTest(path=name):
                self.assertFalse((PACKAGE / name).exists())

    def test_import_ownership(self):
        problems = []
        for path in PACKAGE.rglob("*.py"):
            owner = path.relative_to(PACKAGE).parts[0]
            for target in imports(path):
                forbidden = ["tfts.contracts", "tfts.features", "benchmark", "examples"]
                if owner == "data":
                    forbidden += ["tfts.models", "tfts.tasks", "tfts.training", "tfts.estimator", "tfts.generation"]
                if owner == "models":
                    forbidden += ["tfts.tasks", "tfts.data", "tfts.estimator", "tfts.training"]
                if owner == "tasks":
                    forbidden += ["tfts.training", "tfts.estimator", "tfts.data"]
                if any(target == name or target.startswith(name + ".") for name in forbidden):
                    problems.append(f"{path.relative_to(ROOT)} imports {target}")
        self.assertEqual(problems, [])

    def test_data_stages_import_without_tensorflow(self):
        code = """
import sys
import tfts.data.pipeline
import tfts.data.windows
import tfts.data.features
assert 'tensorflow' not in sys.modules
assert 'tfts.models' not in sys.modules
assert 'tfts.tasks' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, capture_output=True, text=True)

    def test_root_import_is_lazy(self):
        code = """
import sys
import tfts
for name in ('tensorflow', 'pandas', 'torch', 'optuna', 'benchmark', 'tfts.generation'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, capture_output=True, text=True)

    def test_backbones_share_the_core_registry(self):
        from tfts.models.catalog import get_model_class, list_models
        from tfts.registry import registrations

        names = list_models()
        entries = registrations("backbone")
        self.assertEqual(set(entries), set(names))
        for name in names:
            self.assertIs(entries[name].factory, get_model_class(name))
