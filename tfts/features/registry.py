import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class FeatureRegistry:
    def __init__(self) -> None:
        self.columns = []

    def register(self, cols):
        self.columns.extend(cols if isinstance(cols, list) else [cols])

    def get_features(self):
        return self.columns

    def hash_features(self):
        return hashlib.md5(",".join(sorted(self.columns)).encode()).hexdigest()

    def save(self, filepath: str = "feature_columns.json") -> bool:
        with open(filepath, "w") as f:
            json.dump({"features": self.columns}, f, indent=2)

    def __repr__(self) -> str:
        return f"FeatureRegistry('default') with {len(self._feature_names)} features"
