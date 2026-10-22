"""Shared benchmark file utilities."""

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash a data file in bounded chunks on all supported Python versions."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
