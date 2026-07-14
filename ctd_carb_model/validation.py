"""
ctd_carb_model.validation
==========================
PART 3 of the modelling pipeline: leakage-safe validation.

The central rule: split by GROUP, never by row. Rows from the same cast are not
independent (a cast's profile is smooth), so a random row split lets the model
"predict" a depth it effectively already saw. Holding out whole casts/cruises/
seasons forces the model to predict genuinely unseen water — the real task.

This module:
  - provides group splitters (leave-one-cast/station/cruise/season-out),
  - runs a model through a chosen split and collects OUT-OF-GROUP predictions
    (every prediction made by a model that never saw that group in training),
  - reports honest metrics (R2, RMSE, MAE, bias) pooled and per fold,
  - refuses to over-claim: it tells you when the data cannot support a split.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .models import CarbModel


# ---------------------------------------------------------------------------
# Group splitters
# ---------------------------------------------------------------------------


def _group_splits(df: pd.DataFrame, group_column: str) -> Iterator[tuple]:
    if group_column not in df.columns:
        raise KeyError(f"No '{group_column}' column — cannot split on it.")
    for value, group in df.groupby(group_column, dropna=False):
        test_idx = group.index
        train_idx = df.index.difference(test_idx)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx, value


SPLIT_REGISTRY = {
    "leave_one_cast_out": "cast_id",
    "leave_one_station_out": "station",
    "leave_one_cruise_out": "cruise_id",
    "leave_one_season_out": "season",
}


def n_groups(df: pd.DataFrame, split: str) -> int:
    col = SPLIT_REGISTRY.get(split)
    return df[col].nunique() if col and col in df.columns else 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    n = len(a)
    if n < 2:
        return {"n": n, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    err = p - a
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return {
        "n": n,
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


# ---------------------------------------------------------------------------
# The core: run a model through a split, collect out-of-group predictions
# ---------------------------------------------------------------------------


def cross_validate(
    df: pd.DataFrame,
    model_factory,          # callable: () -> a fresh CarbModel
    response: str,
    split: str = "leave_one_cast_out",
) -> dict:
    """Refit a FRESH model on each training fold, predict the held-out group,
    and collect honest out-of-group predictions.

    model_factory must return a NEW model each call (so folds don't share a
    fitted state). Returns a dict with per-row predictions, per-fold metrics,
    and pooled out-of-group metrics.
    """
    if split not in SPLIT_REGISTRY:
        raise KeyError(f"Unknown split {split!r}. Options: {list(SPLIT_REGISTRY)}")
    group_col = SPLIT_REGISTRY[split]
    if group_col not in df.columns:
        return {"response": response, "split": split, "ok": False,
                "note": f"no '{group_col}' column; split not possible."}
    if df[group_col].nunique() < 2:
        return {"response": response, "split": split, "ok": False,
                "note": f"only one {group_col} present; need >=2 groups to validate."}

    rows = []
    fold_metrics = []
    # the predictors this model will use (so we can drop test rows missing them)
    probe = model_factory()
    model_predictors = getattr(probe, "predictors", [])
    for train_idx, test_idx, held in _group_splits(df, group_col):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        try:
            model = model_factory().fit(train, response)
        except (ValueError, RuntimeError):
            continue
        # only score test rows that have the response AND all predictors present
        # (a model must not receive NaN inputs — some models reject them)
        need = [response] + [p for p in model_predictors if p in test.columns]
        test_sub = test.dropna(subset=need)
        if test_sub.empty:
            continue
        try:
            yhat = model.predict(test_sub)
        except (ValueError, RuntimeError):
            continue
        yact = pd.to_numeric(test_sub[response], errors="coerce").to_numpy(float)
        for i in range(len(test_sub)):
            rows.append({"held_out": held, "actual": yact[i], "predicted": float(yhat[i])})
        fold_metrics.append({"held_out": held, **metrics(yact, yhat)})

    if not rows:
        return {"response": response, "split": split, "ok": False,
                "note": "no usable folds (too little data)."}

    preds = pd.DataFrame(rows)
    pooled = metrics(preds["actual"].to_numpy(), preds["predicted"].to_numpy())
    return {
        "response": response,
        "split": split,
        "ok": True,
        "predictions": preds,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "pooled": pooled,
        "n_folds": len(fold_metrics),
    }


# ---------------------------------------------------------------------------
# Honesty helpers
# ---------------------------------------------------------------------------


def coverage(df: pd.DataFrame, split: str) -> pd.DataFrame:
    col = SPLIT_REGISTRY.get(split)
    if not col or col not in df.columns:
        return pd.DataFrame()
    c = df.groupby(col, dropna=False).size().rename("n").reset_index()
    c["fraction"] = (c["n"] / c["n"].sum()).round(3)
    return c.sort_values("n", ascending=False).reset_index(drop=True)


def readiness(df: pd.DataFrame) -> str:
    n = len(df)
    casts = df["cast_id"].nunique() if "cast_id" in df.columns else 0
    cruises = df["cruise_id"].nunique() if "cruise_id" in df.columns else 0
    seasons = df["season"].nunique() if "season" in df.columns else 0
    msg = f"{n} samples, {casts} casts, {cruises} cruise(s), {seasons} season(s). "
    if cruises < 2 and seasons < 2:
        msg += ("Single cruise/season: leave-one-cast-out is the only honest "
                "test available. You CANNOT yet validate temporal or seasonal "
                "transfer — treat any model as exploratory.")
    elif cruises < 3:
        msg += ("Few cruises: leave-one-cruise-out runs but each fold is a "
                "strong test on little data; report per-fold results.")
    else:
        msg += "Enough groups for leave-one-cruise-out and seasonal hold-out."
    return msg
