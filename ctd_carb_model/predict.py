"""
ctd_carb_model.predict
======================
Apply trained models to a full CTD profile to produce CONTINUOUS predicted
carbonate profiles — the input the plotting tool sections into transects.

Workflow:
  1. Train each measured-variable model (pH, TA) on the bottle table (the rows
     with measured carbonate + CTD), using AICc-selected predictors by default.
  2. Apply to the full-profile table (every CTD level) to predict pH and TA
     continuously.
  3. Recompute the derived variables (DIC, pCO2, omega) from the predicted pH+TA
     via PyCO2SYS, so they stay carbonate-system-consistent.
  4. Flag which predicted casts had NO local carbonate to train against (these
     are extrapolation — the model never saw their cruise/region).

Honesty: predicted columns are suffixed '_predicted'; an 'is_extrapolation'
flag marks casts/cruises absent from the training data. With a weak model
(low out-of-group R2) these profiles are EXPLORATORY, not validated fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as _data
from . import models as _models
from . import selection as _selection


def _resolve_on_profile(profile: pd.DataFrame, aliases: list[str]) -> str | None:
    """Resolve a predictor onto the PROFILE table by best coverage.

    The profile-centric table holds the full CTD at every level plus the
    carbonate columns at the sampled levels only. Several aliases can match —
    e.g. 'salinity' (carbonate side, sparse) and 'Salinity_PSU' (CTD side,
    every level). First-match would take the sparse one and silently restrict
    prediction to the sampled rows, so we take the alias with the highest
    non-null coverage instead, breaking ties by alias order.
    """
    best, best_cov = None, -1.0
    for a in aliases:
        if a not in profile.columns:
            continue
        cov = pd.to_numeric(profile[a], errors="coerce").notna().mean()
        if cov > best_cov:
            best, best_cov = a, cov
    return best


def _train_one(train_df: pd.DataFrame, response: str, predictors: list[str],
               select_by: str | None) -> tuple[_models.CarbModel, list[str]]:
    """Fit an MLR for one response, optionally AICc-selecting predictors first."""
    used = predictors
    if select_by:
        try:
            used, _ = _selection.select_predictors(train_df, response, predictors, select_by)
        except (ValueError, KeyError):
            used = predictors
    model = _models.build_model("MLR", used).fit(train_df, response)
    return model, used


def predict_profiles(
    bottle_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    responses: list[str] | None = None,
    predictors: list[str] | None = None,
    select_by: str | None = "aicc",
    recompute_derived: bool = True,
    recompute_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train on the bottle table, predict carbonate at every full-profile level.

    Returns (predicted_profile_df, info). The output is profile_df plus
    '<response>_predicted' columns for the MEASURED responses, recomputed
    derived '*_predicted' columns, and an 'is_extrapolation' flag.
    """
    # prepare the training (bottle) data through the same pipeline
    bottle_ds = _data.prepare(bottle_df)
    predictors = predictors or bottle_ds.predictors
    measured = [r for r in (responses or bottle_ds.responses)
                if bottle_ds.response_status.get(r) == "measured"]
    if not measured:
        raise ValueError("No measured responses (pH/TA) available to train on.")

    # harmonise the profile table's column names to canonical predictor names
    profile = profile_df.copy()
    # resolve predictors onto the profile using the same alias logic as data.prepare
    for canonical in predictors:
        if canonical in profile.columns:
            continue
        src = _resolve_on_profile(
            profile, _data.PREDICTOR_ALIASES.get(canonical, [canonical])
        )
        if src is not None:
            profile[canonical] = profile[src]

    # A profile-centric table carries the CTD at EVERY level but carbonate-side
    # columns only at the sampled levels. If a predictor resolved onto a sparse
    # carbonate column, prediction would be limited to the sampled rows — the
    # opposite of a continuous profile. Fail loudly rather than silently
    # returning a near-empty field.
    sparse = {}
    for canonical in predictors:
        if canonical in profile.columns:
            cov = pd.to_numeric(profile[canonical], errors="coerce").notna().mean()
            if cov < 0.5:
                sparse[canonical] = cov
    if sparse:
        detail = ", ".join(f"{k} ({v:.0%} of levels)" for k, v in sparse.items())
        raise ValueError(
            "Predictor(s) resolved onto sparse column(s) in the profile table: "
            f"{detail}. A continuous profile needs the CTD-native columns "
            "(e.g. Temperature_C, Salinity_PSU, Oxygen_umol_kg, Depth_m), which "
            "are populated at every level. Check the profile table's column "
            "names against PREDICTOR_ALIASES."
        )

    info = {"selected_predictors": {}, "trained_on": {}, "responses": measured}

    # train each measured model and predict on the full profile
    for response in measured:
        model, used = _train_one(bottle_ds.frame, response, predictors, select_by)
        info["selected_predictors"][response] = used
        info["trained_on"][response] = int(len(bottle_ds.frame))
        # predict only where all the model's predictors are present
        mask = profile[used].notna().all(axis=1)
        profile[f"{response}_predicted"] = np.nan
        if mask.any():
            profile.loc[mask, f"{response}_predicted"] = model.predict(profile.loc[mask])

    # recompute derived variables from the predicted pH + TA
    ph_col = "ph_total_predicted"
    ta_col = "total_alkalinity_umol_kg_predicted"
    if recompute_derived and ph_col in profile.columns and ta_col in profile.columns:
        ok = profile[ph_col].notna() & profile[ta_col].notna()
        if ok.any():
            try:
                rec = _models.recompute_carbonate_system(
                    profile.loc[ok], ph_col, ta_col,
                    **(recompute_kwargs or {}))
                for c in rec.columns:
                    # map e.g. dic_umol_kg_recomputed -> dic_umol_kg_predicted
                    canonical = c.replace("_recomputed", "_predicted")
                    profile[canonical] = np.nan
                    profile.loc[ok, canonical] = rec[c].to_numpy()
                info["recomputed"] = [c.replace("_recomputed", "_predicted") for c in rec.columns]
            except RuntimeError as exc:
                info["recompute_error"] = str(exc)

    # flag extrapolation: casts/cruises NOT represented in the training bottles
    train_casts = set(bottle_ds.frame["cast_id"].astype(str)) if "cast_id" in bottle_ds.frame else set()
    train_cruises = set(bottle_ds.frame["cruise_id"].astype(str)) if "cruise_id" in bottle_ds.frame else set()

    # Judge extrapolation on the profile's OWN identity columns, which are
    # populated at every level. The carbonate-side copies (e.g. a 'cruise_id'
    # carried over from the bottles) exist only on sampled rows, so testing
    # them would mark every unsampled level as extrapolation — the opposite of
    # the truth for a cast that does have bottles. Cast is the meaningful unit:
    # a cast with no bottles of its own is what the model never saw.
    cast_col = _resolve_on_profile(profile, ["cast_id", "Cast_ID", "cast"])
    cruise_col = _resolve_on_profile(profile, ["Cruise_ID", "cruise_id", "cruise"])

    if cast_col is not None and train_casts:
        profile["is_extrapolation"] = ~profile[cast_col].astype(str).isin(train_casts)
    elif cruise_col is not None and train_cruises:
        profile["is_extrapolation"] = ~profile[cruise_col].astype(str).isin(train_cruises)
    else:
        profile["is_extrapolation"] = False

    info["n_profile_rows"] = int(len(profile))
    info["n_predicted_rows"] = int(profile.get(ph_col, pd.Series(dtype=float)).notna().sum())
    info["n_extrapolation_rows"] = int(profile["is_extrapolation"].sum())
    info["extrapolated_casts"] = sorted(
        profile.loc[profile["is_extrapolation"], cast_col].astype(str).unique()
    ) if cast_col is not None else []
    return profile, info


def write_predicted_profiles(profile: pd.DataFrame, out_path) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out_path, index=False)
    return out_path


def summarise(info: dict) -> str:
    lines = ["Predicted-profile generation:"]
    for r, preds in info.get("selected_predictors", {}).items():
        lines.append(f"  {r}: predicted from [{', '.join(preds)}] "
                     f"(trained on {info['trained_on'][r]} bottles)")
    if info.get("recomputed"):
        lines.append(f"  recomputed from predicted pH+TA: {', '.join(info['recomputed'])}")
    lines.append(f"  rows predicted: {info.get('n_predicted_rows')} of {info.get('n_profile_rows')}")
    n_ex = info.get("n_extrapolation_rows", 0)
    if n_ex:
        lines.append(f"  EXTRAPOLATION: {n_ex} rows on casts with no local carbonate "
                     f"training data: {', '.join(info.get('extrapolated_casts', [])[:8])}"
                     f"{' ...' if len(info.get('extrapolated_casts', [])) > 8 else ''}")
        lines.append("  -> these predicted casts rest on no nearby samples; treat as "
                     "exploratory extrapolation, not interpolation.")
    return "\n".join(lines)