"""Evaluator: runs metrics over predictions and saves results to CSV."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray

from .metrics import lat_weights, rmse, rmse_per_level, acc, skill_score_rmse, skill_score_acc


class Evaluator:
    """Computes GraphCast paper metrics and serializes to CSV.

    Usage
    -----
    ev = Evaluator(targets, climatology=clim)

    df_gc    = ev.evaluate(predictions_graphcast,  name="graphcast_pretrained")
    df_pers  = ev.evaluate(predictions_persistence, name="persistence")
    df_ft    = ev.evaluate(predictions_finetuned,   name="finetuned_encoder")

    ev.save("results/baselines.csv")
    ev.save_skill_scores("results/skill_scores.csv", model_name="finetuned_encoder",
                          baseline_name="graphcast_pretrained")
    """

    def __init__(
        self,
        targets: xarray.Dataset,
        climatology: Optional[xarray.Dataset] = None,
    ):
        self._targets = targets
        self._climatology = climatology
        self._weights = lat_weights(targets.coords["lat"].values)
        self._records: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: xarray.Dataset,
        name: str,
    ) -> pd.DataFrame:
        """Compute RMSE (and ACC if climatology is available) for one model.

        Returns a DataFrame with columns:
            model, variable, level, lead_time, rmse, acc
        and appends it to the internal records for later saving.
        """
        rows = []

        # ── RMSE (Eq. 20) ────────────────────────────────────────────────────
        rmse_scores = rmse(predictions, self._targets, self._weights)
        for var, leads in rmse_scores.items():
            for lead, val in leads.items():
                rows.append({
                    "model": name, "variable": var, "level": "all",
                    "lead_time": lead, "rmse": val, "acc": np.nan,
                })

        # ── RMSE per level (for 3-D vars) ─────────────────────────────────
        rmse_lev = rmse_per_level(predictions, self._targets, self._weights)
        for var, levels in rmse_lev.items():
            for level, leads in levels.items():
                for lead, val in leads.items():
                    rows.append({
                        "model": name, "variable": var, "level": level,
                        "lead_time": lead, "rmse": val, "acc": np.nan,
                    })

        # ── ACC (Eq. 29) — only if climatology provided ───────────────────
        if self._climatology is not None:
            acc_scores = acc(
                predictions, self._targets, self._climatology, self._weights
            )
            for var, leads in acc_scores.items():
                for lead, val in leads.items():
                    # Update matching row if it exists, else append
                    for row in rows:
                        if (row["model"] == name and row["variable"] == var
                                and row["lead_time"] == lead and row["level"] == "all"):
                            row["acc"] = val
                            break

        self._records.extend(rows)
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Save all evaluated models to a single CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._records)
        df.to_csv(path, index=False)
        print(f"Metrics saved → {path}  ({len(df)} rows)")

    def save_skill_scores(
        self,
        path: str | Path,
        model_name: str,
        baseline_name: str,
    ) -> None:
        """Compute and save normalized skill scores (model vs baseline).

        Skill scores (paper definition):
          RMSE: (RMSE_model - RMSE_baseline) / RMSE_baseline   — negative = model better
          ACC : (ACC_model  - ACC_baseline)  / (1 - ACC_baseline) — positive = model better
        """
        df = pd.DataFrame(self._records)

        model_df    = df[df["model"] == model_name].copy()
        baseline_df = df[df["model"] == baseline_name].copy()

        merge_keys = ["variable", "level", "lead_time"]
        merged = model_df.merge(
            baseline_df[merge_keys + ["rmse", "acc"]],
            on=merge_keys,
            suffixes=("_model", "_baseline"),
        )

        merged["skill_rmse"] = merged.apply(
            lambda r: skill_score_rmse(r["rmse_model"], r["rmse_baseline"]), axis=1
        )
        merged["skill_acc"] = merged.apply(
            lambda r: skill_score_acc(r["acc_model"], r["acc_baseline"])
            if not (np.isnan(r["acc_model"]) or np.isnan(r["acc_baseline"]))
            else np.nan,
            axis=1,
        )

        out = merged[merge_keys + ["rmse_model", "rmse_baseline", "skill_rmse",
                                    "acc_model",  "acc_baseline",  "skill_acc"]]
        out = out.rename(columns={"rmse_model": f"rmse_{model_name}",
                                   "rmse_baseline": f"rmse_{baseline_name}",
                                   "acc_model":  f"acc_{model_name}",
                                   "acc_baseline":  f"acc_{baseline_name}"})

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(f"Skill scores saved → {path}  ({len(out)} rows)")

    def summary(self, model_name: str = None) -> pd.DataFrame:
        """Print a compact summary table (mean RMSE across lead times, per variable)."""
        df = pd.DataFrame(self._records)
        if model_name:
            df = df[df["model"] == model_name]
        df = df[df["level"] == "all"]
        summary = (
            df.groupby(["model", "variable"])[["rmse", "acc"]]
            .mean()
            .round(4)
            .reset_index()
        )
        print(summary.to_string(index=False))
        return summary
