"""Computes metrics from saved predictions — without re-running the model.

Run this after run_baseline.py has saved predictions to results/.

Usage
-----
  python run_metrics.py
  python run_metrics.py --results-dir results
  python run_metrics.py --climatology path/to/clim.nc
"""

import argparse
from pathlib import Path

import xarray

from src.evaluator import Evaluator


RESULTS_DIR = "results"

BASELINES = [
    ("graphcast_pretrained", "preds_graphcast_pretrained.nc"),
    ("persistence",          "preds_persistence.nc"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    p.add_argument("--climatology", type=str, default=None,
                   help="Path to climatology .nc file for ACC computation")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    targets_path = results_dir / "eval_targets.nc"
    if not targets_path.exists():
        raise FileNotFoundError(
            f"{targets_path} not found. Run run_baseline.py first to save predictions."
        )

    print(f"Loading targets from {targets_path} ...")
    targets = xarray.load_dataset(targets_path)

    climatology = None
    if args.climatology:
        climatology = xarray.load_dataset(args.climatology).compute()
        print(f"Climatology loaded from {args.climatology}")

    evaluator = Evaluator(targets, climatology=climatology)

    for name, filename in BASELINES:
        path = results_dir / filename
        if not path.exists():
            print(f"  [{name}] skipped — {path} not found")
            continue
        print(f"\nEvaluating {name} ...")
        preds = xarray.load_dataset(path)
        df = evaluator.evaluate(preds, name=name)
        print(f"  Done. {len(df)} metric rows.")

    evaluator.save(results_dir / "baselines.csv")
    evaluator.save_skill_scores(
        results_dir / "skill_scores_pretrained_vs_persistence.csv",
        model_name="graphcast_pretrained",
        baseline_name="persistence",
    )

    print("\nSummary (mean RMSE across lead times):")
    evaluator.summary()


if __name__ == "__main__":
    main()
