"""Generates and saves baseline metrics before any fine-tuning.

Run this ONCE before training. Results saved to results/baselines.csv
and results/skill_scores.csv.

Baselines computed
------------------
1. graphcast_zero_shot : pre-trained GraphCast, no fine-tuning
2. persistence          : x̂(t+τ) = x(t)

Metrics (GraphCast paper, Verification Methods section)
---------------------------------------------------------
- RMSE  (Eq. 20): latitude-weighted, sqrt INSIDE mean over init times
- ACC   (Eq. 29): latitude-weighted anomaly correlation coefficient
                  (only computed if --climatology is provided)
- Skill scores  : normalized RMSE and ACC differences vs persistence

Usage
-----
  python run_baseline.py
  python run_baseline.py --eval-steps 4   # fewer lead times for quick test
"""

import argparse
import os
import warnings

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
warnings.filterwarnings("ignore")

from src.patch import GraphCastPatcher
from src.data_loader import GCSDataLoader
from src.finetuning import FrozenModuleStrategy
from src.model import GraphCastModel
from src.baselines import PersistenceBaseline, ZeroShotBaseline
from src.evaluator import Evaluator


PARAMS_FILE = (
    "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 "
    "- pressure levels 13 - mesh 2to6 - precipitation output only.npz"
)
RESULTS_DIR = "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-steps", type=int, default=None,
                   help="Number of lead-time steps to evaluate (default: all available)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset filename; defaults to first valid dataset")
    p.add_argument("--climatology", type=str, default=None,
                   help="Path to climatology .nc file for ACC computation")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Patch graphcast.py
    GraphCastPatcher().apply()

    # 2. Load checkpoint
    loader = GCSDataLoader()
    ckpt = loader.load_checkpoint(PARAMS_FILE)
    params, state = ckpt.params, {}
    model_config, task_config = ckpt.model_config, ckpt.task_config

    # 3. Load dataset
    datasets = loader.list_datasets(model_config, task_config)
    dataset_file = args.dataset or datasets[0]
    print(f"Dataset: {dataset_file}")
    example_batch = loader.load_dataset(dataset_file)

    # 4. Split — baselines only need eval split
    eval_steps = args.eval_steps or (example_batch.sizes["time"] - 2)
    (
        _train_inputs, _train_targets, _train_forcings,
        eval_inputs, eval_targets, eval_forcings,
    ) = GCSDataLoader.split_inputs_targets(
        example_batch, task_config,
        train_steps=1,
        eval_steps=eval_steps,
    )
    print(f"Eval: {eval_steps} lead-time steps")
    print(f"  inputs : {dict(eval_inputs.dims)}")
    print(f"  targets: {dict(eval_targets.dims)}")

    # 5. Normalization stats
    diffs_stddev, mean_by_level, stddev_by_level = loader.load_normalization_stats()

    # 6. Build model (strategy doesn't matter for zero-shot — we use all params)
    strategy = FrozenModuleStrategy(preset="decoder")
    model = GraphCastModel(
        model_config=model_config,
        task_config=task_config,
        diffs_stddev_by_level=diffs_stddev,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
        strategy=strategy,
    )

    # 7. Load climatology if provided (needed for ACC)
    climatology = None
    if args.climatology:
        import xarray
        climatology = xarray.load_dataset(args.climatology).compute()
        print(f"Climatology loaded from {args.climatology}")
    else:
        print("No climatology provided — ACC will be skipped. "
              "Pass --climatology <path.nc> to enable it.")

    # 8. Evaluator
    evaluator = Evaluator(eval_targets, climatology=climatology)

    # ── Baseline 1: GraphCast zero-shot ──────────────────────────────────────
    print("\n[1/2] Running GraphCast zero-shot baseline...")
    zero_shot = ZeroShotBaseline(model, params, state)
    preds_zs = zero_shot.predict(eval_inputs, eval_targets, eval_forcings)
    df_zs = evaluator.evaluate(preds_zs, name="graphcast_zero_shot")
    print(f"  Done. {len(df_zs)} metric rows.")

    del preds_zs  # free GPU memory before next prediction

    # ── Baseline 2: Persistence ───────────────────────────────────────────────
    print("\n[2/2] Computing persistence baseline...")
    persistence = PersistenceBaseline()
    preds_pers = persistence.predict(eval_inputs, eval_targets)
    df_pers = evaluator.evaluate(preds_pers, name="persistence")
    print(f"  Done. {len(df_pers)} metric rows.")

    # ── Save results ──────────────────────────────────────────────────────────
    print()
    evaluator.save(f"{RESULTS_DIR}/baselines.csv")
    evaluator.save_skill_scores(
        f"{RESULTS_DIR}/skill_scores_zeroshot_vs_persistence.csv",
        model_name="graphcast_zero_shot",
        baseline_name="persistence",
    )

    print("\nSummary (mean RMSE across lead times):")
    evaluator.summary()

    print(f"\nBaselines complete. Files in ./{RESULTS_DIR}/")
    print("Run run_training.py to fine-tune, then evaluate with:")
    print("  evaluator.evaluate(finetuned_predictions, name='finetuned_encoder')")
    print("  evaluator.save_skill_scores(..., model_name='finetuned_encoder',")
    print("                              baseline_name='graphcast_zero_shot')")


if __name__ == "__main__":
    main()
