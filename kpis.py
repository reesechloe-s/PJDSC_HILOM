"""Scorecards that tie the dashboard to the proposal's stated targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Proposal targets (PJDSC26 503 proposal, Executive Summary / Validation section)
TARGET_WAIT_MIN = 45.0
TARGET_MEAN_WAIT_REDUCTION = 0.10
TARGET_VIOLATION_REDUCTION = 0.15
FULL_CAPACITY = 0.999  # "capacity violation" = sampled bed utilisation at/above this

_PATIENT_WAIT_COLS = ("total_wait", "total_wait_minutes", "wait_total")


def _reduction(base: float, new: float):
    """Relative reduction; None when the baseline is 0 (undefined, not a pass)."""
    if base is None or not np.isfinite(base) or base <= 0:
        return None
    return (base - new) / base


def scenario_kpis(
    hourly: pd.DataFrame,
    patients: pd.DataFrame | None = None,
    capacity_threshold: float = FULL_CAPACITY,
) -> dict:
    """hourly needs `utilization` and `wait_min` (as built by _dashboard_metrics)."""
    mean_wait = float(hourly["wait_min"].mean())
    if patients is not None and not patients.empty:
        for col in _PATIENT_WAIT_COLS:      # prefer patient-weighted mean if available
            if col in patients.columns:
                mean_wait = float(patients[col].dropna().mean())
                break
    return {
        "mean_wait_min": mean_wait,
        "peak_hourly_wait_min": float(hourly["wait_min"].max()),
        "capacity_violation_rate": float((hourly["utilization"] >= capacity_threshold).mean()),
    }


def compare_scenarios(baseline: dict, optimized: dict) -> pd.DataFrame:
    wait_red = _reduction(baseline["mean_wait_min"], optimized["mean_wait_min"])
    viol_red = _reduction(
        baseline["capacity_violation_rate"], optimized["capacity_violation_rate"]
    )

    def status(ok):
        return "n/a (baseline is 0)" if ok is None else ("PASS" if ok else "FAIL")

    rows = [
        ("Mean wait reduction vs baseline", wait_red, TARGET_MEAN_WAIT_REDUCTION,
         None if wait_red is None else wait_red >= TARGET_MEAN_WAIT_REDUCTION),
        ("Capacity-violation reduction vs baseline", viol_red, TARGET_VIOLATION_REDUCTION,
         None if viol_red is None else viol_red >= TARGET_VIOLATION_REDUCTION),
        ("Peak hourly mean wait stays under target (min)",
         optimized["peak_hourly_wait_min"], TARGET_WAIT_MIN,
         optimized["peak_hourly_wait_min"] < TARGET_WAIT_MIN),
    ]
    return pd.DataFrame(
        [{"KPI": n, "Result": r, "Target": t, "Status": status(ok)} for n, r, t, ok in rows]
    )


# ---- forecast accuracy -------------------------------------------------------
def seasonal_naive(series: pd.Series, horizon: int, season: int = 168) -> np.ndarray:
    """Repeat the last full season (168 h = weekly) to forecast `horizon` steps."""
    if len(series) < season:
        raise ValueError("need at least one full season of history")
    last = series.to_numpy()[-season:]
    return np.resize(last, horizon)


def forecast_scorecard(actual, predicted) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must be same non-zero length")
    err = a - p
    nz = a != 0                                   # MAPE is undefined at 0 arrivals
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "WAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(a))) if np.sum(np.abs(a)) else np.nan,
        "MAPE": float(np.mean(np.abs(err[nz] / a[nz]))) if nz.any() else np.nan,
        "MAPE_n_excluded_zero_hours": int((~nz).sum()),
    }


# ---- dashboard helper (moved out of HILOM.py so it can be unit-tested) -------------
def hourly_dashboard_metrics(
    hourly: pd.DataFrame, patients: pd.DataFrame | None, max_occupancy: float
) -> pd.DataFrame:
    """Add utilization, wait_min, saturated and divert_pct columns to the hourly frame."""
    out = hourly.copy().reset_index(drop=True)   # hour k <-> row k, regardless of input index
    out["utilization"] = out["mean_bed_utilization"].fillna(0.0)
    out["wait_min"] = out["mean_total_wait"].fillna(0.0)
    out["saturated"] = out["utilization"] >= max_occupancy
    out["divert_pct"] = 0.0

    needed = {"arrival_time", "acuity", "diversion_accepted"}
    if patients is not None and not patients.empty and needed <= set(patients.columns):
        nu = patients.loc[patients["acuity"] == "non-urgent"]
        if not nu.empty:
            hour = (nu["arrival_time"] // 60).astype(int)
            g = nu.assign(_hour=hour, _div=nu["diversion_accepted"].astype(float)).groupby("_hour")
            pct = 100.0 * g["_div"].sum() / g.size()
            pct = pct[(pct.index >= 0) & (pct.index < len(out))]
            out.loc[pct.index, "divert_pct"] = pct.to_numpy()
    return out