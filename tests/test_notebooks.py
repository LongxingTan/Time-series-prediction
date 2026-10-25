"""Keep published notebooks usable with the installed TFTS API."""

import ast
from importlib import import_module
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NotebookTest(unittest.TestCase):
    def test_notebooks_use_available_package_imports(self):
        notebooks = sorted((ROOT / "docs/source/tutorials").glob("*.ipynb"))
        self.assertTrue(notebooks)
        for path in notebooks:
            cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
            code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
            self.assertTrue(code_cells, path.name)
            for index, cell in enumerate(code_cells):
                with self.subTest(notebook=path.name, cell=index):
                    source = "".join(cell["source"])
                    if source.startswith("%%time\n"):
                        source = source[len("%%time\n") :]
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            self.assertNotEqual(module.split(".")[0], "examples")
                            if module == "tfts" or module.startswith("tfts."):
                                imported = import_module(module)
                                for alias in node.names:
                                    self.assertTrue(hasattr(imported, alias.name), f"{module}.{alias.name}")
                        if isinstance(node, ast.Call):
                            self.assertNotIn(
                                ast.dump(node.func),
                                (
                                    ast.dump(ast.parse("sys.path.insert", mode="eval").body),
                                    ast.dump(ast.parse("sys.path.append", mode="eval").body),
                                ),
                            )
