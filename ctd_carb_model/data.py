"""
ctd_carb_model.data
===================
PART 1 of the modelling pipeline: data preparation.

This module turns the merge tool's bottle-matched master table into a clean,
model-ready frame, and — crucially — keeps track of which carbonate responses
are MEASURED versus CALCULATED. Everything downstream (models, validation,
comparison, reporting) depends on the choices made here, so each one is
documented.

Why this matters scientifically
--------------------------------
In this project pH and TA are measured in the lab; DIC, pCO2, and the omegas
are *calculated* from the measured pH+TA pair by PyCO2SYS. A model that predicts
a calculated variable is predicting a deterministic function of pH and TA, not
an independent measurement. That is legitimate to do, but it must be LABELLED,
and you must never claim a calculated-variable model was "validated against
independent measurements". This module attaches that label so it travels with
the data into every report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Response status — measured vs calculated
# ---------------------------------------------------------------------------


class ResponseStatus(str, Enum):
    MEASURED = "measured"      # direct laboratory measurement (pH, TA)
    CALCULATED = "calculated"  # derived from the measured pair (DIC, pCO2, omega)


# Canonical response name -> (status, list of accepted spellings in the data).
# The merge passes the carbonate pipeline's columns through unchanged, so the
# aliases here cover the oa_pipeline "best"/"calc" names you actually have.
RESPONSE_DEFINITIONS: dict[str, tuple[ResponseStatus, list[str]]] = {
    "ph_total": (ResponseStatus.MEASURED,
                 ["ph_total", "ph_best", "ph_observed", "ph_co2sys"]),
    "total_alkalinity_umol_kg": (ResponseStatus.MEASURED,
                 ["total_alkalinity_umol_kg", "ta_best_umolkg", "ta_umol_kg", "ta"]),
    "dic_umol_kg": (ResponseStatus.CALCULATED,
                 ["dic_umol_kg", "dic_best_umol_kg", "dic_calculated_umol_kg", "dic"]),
    "pco2_uatm": (ResponseStatus.CALCULATED,
                 ["pco2_uatm", "pco2_best_uatm", "pco2_calc_uatm"]),
    "omega_aragonite": (ResponseStatus.CALCULATED,
                 ["omega_aragonite", "omega_aragonite_calc", "omega_ar"]),
    "omega_calcite": (ResponseStatus.CALCULATED,
                 ["omega_calcite", "omega_calcite_calc", "omega_ca"]),
}


# CTD predictors, in the names the merge tool writes them (ctd_* prefix).
# These are the physical variables we regress the carbonate chemistry on.
DEFAULT_PREDICTORS = [
    "ctd_temperature_c",   # temperature: carbonate equilibria + water-mass history
    "ctd_salinity",        # salinity: mixing / water mass; strongest TA control
    "ctd_oxygen",          # oxygen: respiration / remineralisation proxy for pH, DIC
    "ctd_depth_m",         # depth: structural vertical position
]

# Predictor aliases, so the data's column is found regardless of exact spelling.
PREDICTOR_ALIASES = {
    "ctd_temperature_c": ["ctd_temperature_c", "temperature_degC", "temp_insitu",
                          "temperature_insitu_c", "calc_temperature_c"],
    "ctd_salinity": ["ctd_salinity", "salinity_psu", "salinity", "sal", "calc_salinity"],
    "ctd_oxygen": ["ctd_oxygen", "oxygen_umol_kg", "oxygen_umol_l", "dissolved_oxygen_umol_L"],
    "ctd_depth_m": ["ctd_depth_m", "depth_m", "depth_round_m", "depth_teos10_m"],
    "ctd_sigma_theta": ["ctd_sigma_theta", "sigma_theta_kg_m3", "sigma0_kg_m3"],
    "ctd_pressure_dbar": ["ctd_pressure_dbar", "pressure_dbar", "pressure_insitu_dbar"],
}

# Grouping columns used for leakage-safe validation (Part 3).
GROUP_ALIASES = {
    "cast_id": ["cast_id", "cast", "castno"],
    "cruise_id": ["cruise_id", "cruise"],
    "station": ["station", "station_id"],
    "season": ["season", "upwelling_phase"],
}


@dataclass
class DataConfig:
    responses: list[str] = field(default_factory=lambda: list(RESPONSE_DEFINITIONS.keys()))
    predictors: list[str] = field(default_factory=lambda: list(DEFAULT_PREDICTORS))
    # only use rows the carbonate pipeline judged analysis-ready, if that flag
    # is present (keeps QC-failed rows out of the model)
    require_analysis_ready: bool = True
    analysis_status_column: str = "analysis_audit_status"
    analysis_ready_values: tuple = ("PASS", "REVIEW")  # exclude FAIL by default
    # drop predictors present in fewer than this fraction of rows (a near-empty
    # predictor like a sparse oxygen channel otherwise collapses complete-case n)
    min_predictor_coverage: float = 0.5


@dataclass
class ModelDataset:
    """A model-ready dataset plus the metadata models and reports need."""
    frame: pd.DataFrame
    predictors: list[str]              # resolved predictor columns present
    responses: list[str]               # resolved response columns present
    response_status: dict[str, str]    # response -> 'measured' | 'calculated'
    group_columns: dict[str, str]      # logical -> actual column name
    n_rows: int
    notes: list[str]


def _resolve(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cols = {c.strip(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
    return None


def prepare(df: pd.DataFrame, config: DataConfig | None = None) -> ModelDataset:
    """Turn a merged master table into a model-ready ModelDataset.

    Steps, in order:
      1. resolve predictor columns (rename to canonical ctd_* names)
      2. resolve response columns and record each one's measured/calculated status
      3. resolve grouping columns for later leakage-safe validation
      4. optionally keep only analysis-ready rows
      5. report what was found, what was missing, and any caveats
    """
    config = config or DataConfig()
    out = df.copy()
    notes: list[str] = []

    # 1. predictors -------------------------------------------------------
    resolved_predictors: list[str] = []
    for canonical in config.predictors:
        src = _resolve(out, PREDICTOR_ALIASES.get(canonical, [canonical]))
        if src is None:
            notes.append(f"predictor '{canonical}' not found — skipped.")
            continue
        if src != canonical:
            out[canonical] = out[src]
        resolved_predictors.append(canonical)
    if not resolved_predictors:
        raise ValueError("No predictors resolved. Check the merge output has "
                         "ctd_* columns (e.g. ctd_temperature_c, ctd_salinity).")

    # 2. responses + status ----------------------------------------------
    resolved_responses: list[str] = []
    response_status: dict[str, str] = {}
    for canonical in config.responses:
        if canonical not in RESPONSE_DEFINITIONS:
            notes.append(f"response '{canonical}' has no definition — skipped.")
            continue
        status, aliases = RESPONSE_DEFINITIONS[canonical]
        src = _resolve(out, aliases)
        if src is None:
            notes.append(f"response '{canonical}' not found — skipped.")
            continue
        if src != canonical:
            out[canonical] = out[src]
        resolved_responses.append(canonical)
        response_status[canonical] = status.value
        if status is ResponseStatus.CALCULATED:
            notes.append(
                f"response '{canonical}' is CALCULATED from measured pH+TA — "
                "model predicts a derived quantity, not an independent measurement."
            )
    if not resolved_responses:
        raise ValueError("No responses resolved. Check the merge output has "
                         "pH/TA columns (e.g. ph_best, ta_best_umolkg).")

    # 3. grouping columns -------------------------------------------------
    group_columns: dict[str, str] = {}
    for logical, aliases in GROUP_ALIASES.items():
        src = _resolve(out, aliases)
        if src is not None:
            if src != logical:
                out[logical] = out[src]
            group_columns[logical] = logical

    # 4. analysis-ready filter -------------------------------------------
    if config.require_analysis_ready and config.analysis_status_column in out.columns:
        before = len(out)
        out = out[out[config.analysis_status_column].isin(config.analysis_ready_values)]
        dropped = before - len(out)
        if dropped:
            notes.append(
                f"kept {len(out)}/{before} rows with "
                f"{config.analysis_status_column} in {config.analysis_ready_values} "
                f"(dropped {dropped} FAIL/other)."
            )

    # 5. only rows that actually have a CTD match are usable as predictors
    if "flag_no_ctd_match" in out.columns:
        before = len(out)
        out = out[~out["flag_no_ctd_match"].astype(bool)]
        if before - len(out):
            notes.append(f"dropped {before - len(out)} rows with no CTD match "
                         "(no predictors available for them).")

    # 6. drop predictors with too little coverage (a near-empty column like a
    #    sparse oxygen channel otherwise collapses the complete-case row count)
    kept_predictors = []
    for p in resolved_predictors:
        cov = pd.to_numeric(out[p], errors="coerce").notna().mean() if len(out) else 0.0
        if cov >= config.min_predictor_coverage:
            kept_predictors.append(p)
        else:
            notes.append(
                f"predictor '{p}' dropped — present in only {cov:.0%} of rows "
                f"(< {config.min_predictor_coverage:.0%}); too sparse to use. "
                "Collect more of this variable, or it stays out of the models."
            )
    if not kept_predictors:
        raise ValueError(
            "All predictors fell below the coverage threshold. Your CTD "
            "predictor columns are too sparse in the matched data to model. "
            f"Coverage: " + ", ".join(
                f"{p}={pd.to_numeric(out[p], errors='coerce').notna().mean():.0%}"
                for p in resolved_predictors))
    resolved_predictors = kept_predictors

    return ModelDataset(
        frame=out.reset_index(drop=True),
        predictors=resolved_predictors,
        responses=resolved_responses,
        response_status=response_status,
        group_columns=group_columns,
        n_rows=len(out),
        notes=notes,
    )


def summarise(ds: ModelDataset) -> str:
    """A short, honest description of the model-ready dataset."""
    lines = [
        f"Model-ready dataset: {ds.n_rows} rows.",
        f"Predictors ({len(ds.predictors)}): {', '.join(ds.predictors)}",
        "Responses:",
    ]
    for r in ds.responses:
        lines.append(f"  - {r}  [{ds.response_status[r]}]")
    if ds.group_columns:
        lines.append(f"Grouping available: {', '.join(ds.group_columns)}")
    n_casts = ds.frame["cast_id"].nunique() if "cast_id" in ds.frame.columns else 0
    n_cruises = ds.frame["cruise_id"].nunique() if "cruise_id" in ds.frame.columns else 0
    lines.append(f"Coverage: {n_casts} casts, {n_cruises} cruise(s).")
    if ds.notes:
        lines.append("Notes:")
        lines += [f"  • {n}" for n in ds.notes]
    return "\n".join(lines)
