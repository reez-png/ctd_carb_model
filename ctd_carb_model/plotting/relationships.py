"""
ctd_carb_analysis.relationships
===============================
Exploratory relationship analysis over the bottle-matched table: correlations
between carbonate response variables and CTD predictors, plus depth/station/
season summaries. This is Stage 1 ("observed relationship analysis") in the
project design — it answers "do coherent relationships exist?" before any model
is asked to predict.

All of this operates on MEASURED and matched values only. Nothing here predicts
or interpolates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Default response (carbonate) and predictor (CTD) variables.
DEFAULT_RESPONSES = [
    "ph_total", "total_alkalinity_umol_kg", "dic_umol_kg",
    "pco2_uatm", "omega_aragonite", "omega_calcite",
]
DEFAULT_PREDICTORS = [
    "ctd_temperature_c", "ctd_salinity", "ctd_oxygen",
    "ctd_depth_m", "ctd_sigma_theta",
]


@dataclass
class RelationshipConfig:
    responses: list[str] = field(default_factory=lambda: list(DEFAULT_RESPONSES))
    predictors: list[str] = field(default_factory=lambda: list(DEFAULT_PREDICTORS))
    method: str = "spearman"   # spearman is robust to nonlinearity & outliers
    group_column: str | None = None  # e.g. "upwelling_phase" or "season"


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def correlation_matrix(
    matched: pd.DataFrame, config: RelationshipConfig | None = None
) -> pd.DataFrame:
    """Response x predictor correlation matrix (long form).

    Returns a tidy frame: response, predictor, method, n, correlation.
    Only rows where both variables are present contribute to each pair, so a
    sparse response does not silently drag down unrelated pairs.
    """
    config = config or RelationshipConfig()
    responses = _present(matched, config.responses)
    predictors = _present(matched, config.predictors)

    rows = []
    for r in responses:
        for p in predictors:
            pair = matched[[r, p]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(pair)
            corr = (
                pair[r].corr(pair[p], method=config.method) if n >= 3 else float("nan")
            )
            rows.append(
                {"response": r, "predictor": p, "method": config.method,
                 "n": n, "correlation": corr}
            )
    return pd.DataFrame(rows)


def grouped_correlations(
    matched: pd.DataFrame, config: RelationshipConfig | None = None
) -> pd.DataFrame:
    """Correlation matrix computed separately within each group (e.g. upwelling
    vs non-upwelling). Returns the long-form matrix with a `group` column.
    Falls back to a single 'all' group if no group column is configured/present.
    """
    config = config or RelationshipConfig()
    gcol = config.group_column
    if not gcol or gcol not in matched.columns:
        out = correlation_matrix(matched, config)
        out.insert(0, "group", "all")
        return out

    frames = []
    for gval, gdf in matched.groupby(gcol, dropna=False):
        cm = correlation_matrix(gdf, config)
        cm.insert(0, "group", gval)
        frames.append(cm)
    return pd.concat(frames, ignore_index=True)


def depth_station_summary(
    matched: pd.DataFrame,
    config: RelationshipConfig | None = None,
    depth_column: str = "depth_round_m",
    station_column: str = "station_name",
) -> pd.DataFrame:
    """Per station x nominal-depth summary (mean/std/n) of each response."""
    config = config or RelationshipConfig()
    responses = _present(matched, config.responses)
    keys = [c for c in (station_column, depth_column) if c in matched.columns]
    if not keys or not responses:
        return pd.DataFrame()
    agg = {r: ["mean", "std", "count"] for r in responses}
    out = matched.groupby(keys, dropna=False).agg(agg)
    out.columns = [f"{var}_{stat}" for var, stat in out.columns]
    return out.reset_index()


def season_summary(
    matched: pd.DataFrame,
    config: RelationshipConfig | None = None,
    season_column: str = "season",
) -> pd.DataFrame:
    """Per-season (or per-group) summary of each response variable."""
    config = config or RelationshipConfig()
    col = season_column if season_column in matched.columns else config.group_column
    responses = _present(matched, config.responses)
    if not col or col not in matched.columns or not responses:
        return pd.DataFrame()
    agg = {r: ["mean", "std", "count"] for r in responses}
    out = matched.groupby(col, dropna=False).agg(agg)
    out.columns = [f"{var}_{stat}" for var, stat in out.columns]
    return out.reset_index()
