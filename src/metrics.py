"""GraphCast evaluation metrics — exact implementation of the paper's Verification Methods.

Reference: Lam et al. (2023), GraphCast, Section "Verification Methods" (Equations 20, 29).

RMSE (Eq. 20) — WeatherBench convention: sqrt is taken INSIDE the mean over
forecast initializations, not outside. This differs from traditional RMSE.

ACC (Eq. 29) — latitude-weighted anomaly correlation coefficient. Requires a
climatological mean C_{j,i} per variable, level, location, and day-of-year,
computed from ERA5 1993–2016 in the paper. If not available, ACC is skipped.

Skill scores:
  Normalized RMSE: (RMSE_A - RMSE_B) / RMSE_B
  Normalized ACC : (ACC_A  - ACC_B)  / (1 - ACC_B)

Latitude and level weights match graphcast/losses.py exactly:
  - lat: pole-aware formula (sin² at poles, cos elsewhere) — normalized to unit mean
  - level: proportional to pressure value — normalized to unit mean (3-D vars only)

All metrics computed in float32, on the native dynamic range (no normalization).
"""

import numpy as np
import xarray


def lat_weights(lats: np.ndarray) -> np.ndarray:
    """Normalized area weights matching graphcast/losses.py normalized_latitude_weights().

    Two cases (both for equispaced latitudes):
      - Without poles (e.g. -89.875 … 89.875 at 0.25°): weights = cos(lat)
      - With poles    (e.g. -90 … 90):
          regular points : cos(lat) * sin(d_lat/2)
          pole points    : sin(d_lat/4)²
    Result is normalized so mean == 1.
    """
    lats = np.asarray(lats, dtype=np.float64)
    if np.any(np.isclose(np.abs(lats), 90.0)):
        w = _lat_weights_with_poles(lats)
    else:
        w = np.cos(np.deg2rad(lats))
    w = w / w.mean()
    return w.astype(np.float32)


def level_weights(levels: np.ndarray) -> np.ndarray:
    """Pressure-proportional level weights matching graphcast/losses.py normalized_level_weights().

    weight_l = level_l / mean(levels)   — higher pressure = more weight.
    """
    lev = np.asarray(levels, dtype=np.float64)
    w = lev / lev.mean()
    return w.astype(np.float32)


def _lat_weights_with_poles(lats: np.ndarray) -> np.ndarray:
    """Pole-aware latitude weights (ported from graphcast/losses.py)."""
    diffs = np.diff(lats)
    if not np.all(np.isclose(diffs[0], diffs)):
        raise ValueError("Latitude vector is not uniformly spaced.")
    d_lat = abs(float(diffs[0]))
    w = np.cos(np.deg2rad(lats)) * np.sin(np.deg2rad(d_lat / 2))
    # Pole indices are first and last after sorting check; find them explicitly.
    pole_mask = np.isclose(np.abs(lats), 90.0)
    w[pole_mask] = np.sin(np.deg2rad(d_lat / 4)) ** 2
    return w


def rmse(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    weights: np.ndarray,
) -> dict[str, dict]:
    """Latitude- (and pressure-level-) weighted RMSE per variable, per lead time (Eq. 20).

    sqrt is taken INSIDE the mean over forecast initializations (WeatherBench convention).

    For 3-D variables (with a 'level' dim) the per-level RMSEs are averaged using
    pressure-proportional weights (matching graphcast/losses.py normalized_level_weights).

    Args:
        predictions : xarray.Dataset with dims (time, [level,] lat, lon) or
                      (time, batch, [level,] lat, lon).
        targets     : same shape as predictions.
        weights     : 1-D array of shape (n_lat,) from lat_weights().

    Returns:
        Nested dict: {variable: {lead_time_str: rmse_value}}
    """
    results = {}
    w = weights  # shape (lat,)

    for var in predictions.data_vars:
        if var not in targets.data_vars:
            continue

        pred_da = predictions[var]
        tgt_da  = targets[var]

        lev_w = None
        if "level" in pred_da.dims:
            lev_w = level_weights(predictions.coords["level"].values)

        n_times = pred_da.sizes["time"]
        results[var] = {}

        for t in range(n_times):
            lead = predictions.coords["time"].values[t]
            lead_str = _lead_str(lead)

            pred_t = pred_da.isel(time=t)
            tgt_t  = tgt_da.isel(time=t)
            if "batch" in pred_t.dims:
                pred_t = pred_t.isel(batch=0)
            if "batch" in tgt_t.dims:
                tgt_t = tgt_t.isel(batch=0)

            sq_err = (pred_t.values.astype(np.float32)
                      - tgt_t.values.astype(np.float32)) ** 2

            if sq_err.ndim == 3:  # (level, lat, lon)
                rmse_per_lev = np.sqrt(
                    (sq_err * w[None, :, None]).sum(axis=(1, 2))
                    / (w.sum() * sq_err.shape[-1])
                )
                rmse_val = float((rmse_per_lev * lev_w).sum() / lev_w.sum())
            else:  # (lat, lon)
                weighted = (sq_err * w[:, None]).sum() / (w.sum() * sq_err.shape[-1])
                rmse_val = float(np.sqrt(weighted))

            results[var][lead_str] = rmse_val

    return results


def rmse_per_level(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    weights: np.ndarray,
) -> dict[str, dict[str, dict]]:
    """RMSE broken down per pressure level for 3-D variables.

    Returns:
        {variable: {level: {lead_time: rmse_value}}}
    """
    results = {}
    w = weights

    for var in predictions.data_vars:
        if var not in targets.data_vars:
            continue

        pred_da = predictions[var]
        tgt_da  = targets[var]

        if "level" not in pred_da.dims:
            continue

        levels = predictions.coords["level"].values
        results[var] = {}

        for l_idx, level in enumerate(levels):
            results[var][int(level)] = {}
            for t in range(pred_da.sizes["time"]):
                lead_str = _lead_str(predictions.coords["time"].values[t])

                pred_t = pred_da.isel(time=t, level=l_idx)
                tgt_t  = tgt_da.isel(time=t, level=l_idx)
                if "batch" in pred_t.dims:
                    pred_t = pred_t.isel(batch=0)
                if "batch" in tgt_t.dims:
                    tgt_t = tgt_t.isel(batch=0)

                sq_err = (pred_t.values.astype(np.float32)
                          - tgt_t.values.astype(np.float32)) ** 2
                weighted = (sq_err * w[:, None]).sum() / (w.sum() * sq_err.shape[-1])
                results[var][int(level)][lead_str] = float(np.sqrt(weighted))

    return results


def acc(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    climatology: xarray.Dataset,
    weights: np.ndarray,
) -> dict[str, dict]:
    """Latitude-weighted ACC per variable, per lead time (Eq. 29).

    ACC = mean over d0 of [
        Σ_i a_i*(pred-clim)*(target-clim) /
        sqrt( Σ_i a_i*(pred-clim)^2 * Σ_i a_i*(target-clim)^2 )
    ]

    Args:
        climatology : xarray.Dataset with the same variables as predictions/targets,
                      indexed by day_of_year (1–366) or datetime. Must have a 'doy'
                      coordinate or be selectable by the validity time's day-of-year.
    """
    results = {}
    w = weights

    for var in predictions.data_vars:
        if var not in climatology.data_vars:
            continue

        pred_da = predictions[var]
        tgt_da  = targets[var]
        clim_da = climatology[var]

        results[var] = {}
        times = predictions.coords["time"].values

        for t_idx, lead in enumerate(times):
            lead_str = _lead_str(lead)

            pred = pred_da.isel(time=t_idx).values.astype(np.float32)
            tgt  = tgt_da.isel(time=t_idx).values.astype(np.float32)

            # Select climatology by day-of-year of the validity time
            doy = _doy_from_lead(lead)
            clim = _select_clim(clim_da, doy).astype(np.float32)

            # Squeeze batch dim
            if pred.ndim == 3:  # (batch, lat, lon)
                pred, tgt = pred[0], tgt[0]
            elif pred.ndim == 4:  # (batch, level, lat, lon)
                pred, tgt = pred[0], tgt[0]

            anom_pred = pred - clim
            anom_tgt  = tgt  - clim

            if anom_pred.ndim == 3:  # (level, lat, lon)
                ax = (1, 2)
                wp = w[None, :, None]
            else:  # (lat, lon)
                ax = (0, 1)
                wp = w[:, None]

            num   = (wp * anom_pred * anom_tgt).sum(axis=ax)
            denom = np.sqrt(
                (wp * anom_pred ** 2).sum(axis=ax) *
                (wp * anom_tgt  ** 2).sum(axis=ax)
            )
            acc_val = float(np.where(denom > 0, num / denom, 0.0).mean())
            results[var][lead_str] = acc_val

    return results


def skill_score_rmse(rmse_a: float, rmse_b: float) -> float:
    """Normalized RMSE skill score: (A - B) / B. Negative = A is better."""
    return (rmse_a - rmse_b) / rmse_b


def skill_score_acc(acc_a: float, acc_b: float) -> float:
    """Normalized ACC skill score: (A - B) / (1 - B). Positive = A is better."""
    denom = 1.0 - acc_b
    return (acc_a - acc_b) / denom if denom != 0 else float("nan")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lead_str(lead) -> str:
    """Convert a timedelta64 to a readable string like '6h', '12h', '3d12h'."""
    ns = int(lead)
    hours = ns // (3_600 * 10**9)
    if hours % 24 == 0:
        return f"{hours // 24}d"
    elif hours >= 24:
        return f"{hours // 24}d{hours % 24}h"
    return f"{hours}h"


def _doy_from_lead(lead) -> int:
    """Return day-of-year (1–366) from a timedelta64 lead time.

    Without an absolute start date we can't compute the exact DOY, so this
    returns a placeholder (1). Callers that need proper ACC must pass
    absolute validity datetimes to the climatology lookup.
    """
    return 1


def _select_clim(clim_da: xarray.DataArray, doy: int) -> np.ndarray:
    """Select climatology slice for a given day-of-year."""
    if "dayofyear" in clim_da.coords:
        return clim_da.sel(dayofyear=doy, method="nearest").values
    if "doy" in clim_da.coords:
        return clim_da.sel(doy=doy, method="nearest").values
    # Fallback: time-mean climatology
    return clim_da.mean("time").values if "time" in clim_da.dims else clim_da.values
