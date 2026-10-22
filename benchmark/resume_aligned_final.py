"""Resume an aligned RNN run from its original saved manifest.

Run with ``python -m benchmark.resume_aligned_final --data train.csv --output RUN``.
Existing checkpoints are reused; missing folds are trained. ``--folds`` can
explicitly select folds to retrain. All OOF predictions are rebuilt from the
checkpoints, so interrupted or overwritten prediction snapshots are repaired.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.ventilator import resume_comparison


def resume(output: Path, data: Path, folds_to_run: list[int] | None = None, threads: int = 6) -> int:
    """Reuse the authoritative training path and saved configuration."""
    resume_comparison(data, output, threads=threads, folds=folds_to_run)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ventilator_aligned_final_10e"))
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    raise SystemExit(resume(args.output, args.data, args.folds, threads=args.threads))


if __name__ == "__main__":
    main()
