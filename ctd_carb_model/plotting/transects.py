"""
ctd_carb_analysis.transects
===========================
SCAFFOLD for 2D (and later 3D) gridded carbonate fields along a transect — the
"square-pixel section with bathymetry" product.

  ⚠️ READ THIS FIRST. A gridded transect of pH / TA / Ω is a PREDICTED,
  interpolated field. Between your sampled depths and between your casts, the
  values are estimated, not measured. This product therefore belongs entirely
  to the model branch, must be built on a VALIDATED model (Steps 04–06), and
  must be labelled as model output — never as a measured section.

What a finished transect product looks like:

    distance along transect (km)  →   x-axis
    depth (m)                     →   y-axis (increasing downward)
    pixel value                   →   predicted pH_T / TA / Ω at that (x, z)
    bathymetry                    →   a masked region below the seafloor

How it is built (the honest pipeline):

  1. Order the casts along the transect and compute along-track distance.
  2. For each cast, predict a continuous 1-m carbonate profile from the CTD
     profile using the validated model (Step 06 output).
  3. Interpolate those predicted profiles onto a regular (distance × depth)
     grid — this is the step that invents values between casts, so its
     uncertainty must be carried and shown.
  4. Mask grid cells below the seafloor using a bathymetry source (e.g. a
     GEBCO netCDF you download), so the section does not show "water" inside
     the seabed.
  5. Plot as a pcolormesh (square pixels), overlay sample locations, and label
     it as a model-predicted section with the standard disclaimer.

This module provides the scaffolding and the interpolation/masking helpers.
The heavier pieces (reading a bathymetry netCDF, full 3D volume assembly) are
marked as TODO entry points with guidance, because they depend on the exact
bathymetry product you download and on having a validated model first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import PREDICTED_PROFILE_DISCLAIMER


@dataclass
class TransectConfig:
    cast_column: str = "cast_id"
    station_column: str = "station"   # the fixed-location label, e.g. J1, J2, J3
    lat_column: str = "latitude"
    lon_column: str = "longitude"
    depth_column: str = "ctd_depth_m"
    # grid resolution
    distance_bin_km: float = 1.0
    depth_bin_m: float = 1.0
    # which predicted variable(s) to grid
    value_columns: list[str] = field(
        default_factory=lambda: ["ph_total_predicted", "total_alkalinity_umol_kg_predicted"]
    )
    # A transect is an ORDERED list of stations, e.g. ["J1", "J2", "J3"].
    # The x-axis is real along-track distance between them (haversine).
    transect_stations: list[str] | None = None
    # If a station has several casts (multiple cruises), pick which cruise to
    # section; None = use all rows present for that station (assumes one
    # occupation in the supplied predicted-profile file).
    cruise_column: str = "cruise_id"
    cruise: str | None = None
    # legacy: order casts by geographic distance from the first if no stations
    cast_order: list[str] | None = None


# ---------------------------------------------------------------------------
# Step 1 — along-track distance
# ---------------------------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def cast_positions(
    predicted_profiles: pd.DataFrame, config: TransectConfig | None = None
) -> pd.DataFrame:
    """One row per transect point (station, or cast if no station column) with
    lat/lon and cumulative along-track distance in km.

    Station mode (recommended for J1, J2, J3 ...):
      - if `config.transect_stations` is given, only those stations are used,
        IN THAT ORDER, and the x-axis distance is the real haversine distance
        between consecutive stations.
      - if `config.cruise` is given, only that cruise's occupation is used
        (a station may be occupied on several cruises).

    Cast mode (fallback): if there is no station column, points are casts,
    optionally ordered by `config.cast_order`, else by geographic distance.
    """
    config = config or TransectConfig()
    df = predicted_profiles

    # pick the grouping label: station if present, else cast
    use_station = config.station_column in df.columns
    label_col = config.station_column if use_station else config.cast_column

    need = [label_col, config.lat_column, config.lon_column]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(
            f"Transect needs columns {missing}. Station mode needs "
            f"'{config.station_column}', latitude, longitude; these come from "
            "the Step-06 cast-log merge. Add a station label to your data."
        )

    # optional: restrict to one cruise/occupation
    if config.cruise is not None and config.cruise_column in df.columns:
        df = df[df[config.cruise_column].astype(str) == str(config.cruise)]
        if df.empty:
            raise ValueError(f"No rows for cruise {config.cruise!r}.")
    elif config.cruise is None and config.cruise_column in df.columns:
        # Guard: if any station was occupied on more than one cruise and the
        # caller did NOT pick one, refuse — otherwise we would silently average
        # different occupations (e.g. different months) into one pixel.
        per_station_cruises = (
            df.groupby(label_col)[config.cruise_column].nunique()
        )
        multi = per_station_cruises[per_station_cruises > 1]
        if len(multi) > 0:
            raise ValueError(
                "Stations occupied on multiple cruises were found "
                f"({', '.join(map(str, multi.index[:5]))}"
                f"{' ...' if len(multi) > 5 else ''}), but no `cruise` was "
                "specified. Sectioning them together would average different "
                "occupations (e.g. different months) into single pixels. "
                "Set config.cruise=<one cruise>, or use "
                "grid_transect_2d_per_cruise() to produce one section per cruise."
            )

    # one representative position per label (mean lat/lon over its rows)
    points = (df.groupby(label_col, dropna=False)
              .agg({config.lat_column: "mean", config.lon_column: "mean"})
              .reset_index())

    # ordering
    if use_station and config.transect_stations:
        order = {s: i for i, s in enumerate(config.transect_stations)}
        points = points[points[label_col].isin(order)].copy()
        missing_stn = [s for s in config.transect_stations
                       if s not in set(points[label_col])]
        if missing_stn:
            raise ValueError(
                f"Transect stations not found in data: {missing_stn}. "
                f"Available: {sorted(points[label_col].unique())}"
            )
        points["__order"] = points[label_col].map(order)
        points = points.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    elif not use_station and config.cast_order:
        order = {c: i for i, c in enumerate(config.cast_order)}
        points["__order"] = points[label_col].map(order)
        points = points.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # cumulative haversine distance along the ordered points
    dist = [0.0]
    for i in range(1, len(points)):
        d = haversine_km(
            points[config.lat_column].iloc[i - 1], points[config.lon_column].iloc[i - 1],
            points[config.lat_column].iloc[i], points[config.lon_column].iloc[i],
        )
        dist.append(dist[-1] + d)
    points["distance_km"] = dist
    points = points.rename(columns={label_col: "transect_point"})
    points["point_label"] = points["transect_point"]
    return points


# ---------------------------------------------------------------------------
# Step 3 — grid the predicted profiles onto (distance × depth)
# ---------------------------------------------------------------------------


def grid_transect_2d(
    predicted_profiles: pd.DataFrame,
    value_column: str,
    config: TransectConfig | None = None,
) -> dict:
    """Interpolate predicted per-cast profiles onto a regular distance×depth
    grid. Returns a dict with the grid axes and the value matrix (NaN where no
    data supports a cell yet — masking by bathymetry happens separately).

    NOTE: this interpolates between casts, inventing values in the gaps. That is
    the defining property of a section product and the reason it must be labelled
    predicted. Linear interpolation here is the simplest defensible choice;
    objective mapping / kriging with a length scale is a later refinement.
    """
    config = config or TransectConfig()
    if value_column not in predicted_profiles.columns:
        raise KeyError(f"{value_column!r} not in predicted profiles. Generate "
                       "predicted profiles (Step 06) first.")
    positions = cast_positions(predicted_profiles, config)

    # the label each profile row is grouped by (station if present, else cast)
    use_station = config.station_column in predicted_profiles.columns
    label_col = config.station_column if use_station else config.cast_column
    point_dist = dict(zip(positions["transect_point"], positions["distance_km"]))

    # restrict to the same cruise if one was selected, and to the transect points
    pp = predicted_profiles.copy()
    if config.cruise is not None and config.cruise_column in pp.columns:
        pp = pp[pp[config.cruise_column].astype(str) == str(config.cruise)]
    pp["distance_km"] = pp[label_col].map(point_dist)
    pp = pp.dropna(subset=["distance_km", config.depth_column, value_column])

    # target grid
    dmax = pp["distance_km"].max()
    zmax = pp[config.depth_column].max()
    xs = np.arange(0, dmax + config.distance_bin_km, config.distance_bin_km)
    zs = np.arange(0, zmax + config.depth_bin_m, config.depth_bin_m)
    grid = np.full((len(zs), len(xs)), np.nan)

    # simple approach: at each grid depth, interpolate across distance using the
    # casts that have a value at (closest) that depth. For dense 1-m predicted
    # profiles every cast has every depth, so this is well posed.
    for zi, z in enumerate(zs):
        # nearest profile row per cast at this depth
        at_z = pp[np.abs(pp[config.depth_column] - z) <= config.depth_bin_m / 2]
        if at_z.empty:
            continue
        per_cast = at_z.groupby("distance_km")[value_column].mean()
        if len(per_cast) < 2:
            # only one cast supports this depth -> fill that column only
            for dkm, val in per_cast.items():
                xi = int(round(dkm / config.distance_bin_km))
                if 0 <= xi < len(xs):
                    grid[zi, xi] = val
            continue
        xp = per_cast.index.to_numpy()
        fp = per_cast.to_numpy()
        order = np.argsort(xp)
        grid[zi, :] = np.interp(xs, xp[order], fp[order], left=np.nan, right=np.nan)

    return {
        "distance_km": xs,
        "depth_m": zs,
        "values": grid,
        "value_column": value_column,
        "cast_positions": positions,
        "disclaimer": PREDICTED_PROFILE_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Step 4 — bathymetry mask (TODO entry point)
# ---------------------------------------------------------------------------


def grid_transect_2d_per_cruise(
    predicted_profiles: pd.DataFrame,
    value_column: str,
    config: TransectConfig | None = None,
) -> dict[str, dict]:
    """Produce one transect grid per cruise/occupation (e.g. one per month).

    Returns a dict mapping cruise -> grid (the same structure grid_transect_2d
    returns). Each cruise is sectioned independently, so different occupations
    of the same station are never averaged together. Cruises that do not have
    all the requested transect stations are skipped with a recorded note.
    """
    config = config or TransectConfig()
    if config.cruise_column not in predicted_profiles.columns:
        # no cruise concept; just return a single grid keyed "all"
        return {"all": grid_transect_2d(predicted_profiles, value_column, config)}

    out: dict[str, dict] = {}
    for cruise in sorted(predicted_profiles[config.cruise_column].dropna().astype(str).unique()):
        cfg_c = dataclasses_replace(config, cruise=cruise)
        try:
            out[cruise] = grid_transect_2d(predicted_profiles, value_column, cfg_c)
        except ValueError as exc:
            # e.g. a cruise that did not occupy every requested station
            out[cruise] = {"error": str(exc), "value_column": value_column}
    return out


def plot_transect_2d_per_cruise(
    grids_by_cruise: dict[str, dict],
    out_dir,
    value_column: str,
    title_prefix: str = "Predicted",
    cmap: str = "viridis",
) -> list:
    """Plot one section figure per cruise from grid_transect_2d_per_cruise().
    Skips cruises that errored (e.g. missing stations). Returns written paths."""
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    written = []
    for cruise, grid in grids_by_cruise.items():
        if "error" in grid:
            continue
        path = plot_transect_2d(
            grid, out_dir / f"section_{value_column}_{cruise}.png",
            title=f"{title_prefix} {value_column} — {cruise}", cmap=cmap,
        )
        written.append(path)
    return written


def load_bathymetry_along_transect(
    netcdf_path: str | Path,
    cast_positions_df: pd.DataFrame,
    config: TransectConfig | None = None,
):  # pragma: no cover - depends on downloaded product
    """TODO: read a bathymetry netCDF (e.g. GEBCO) and return seafloor depth at
    each transect distance.

    Implementation guidance when you have the file:
      - open with xarray: `ds = xarray.open_dataset(netcdf_path)`
      - GEBCO stores elevation in a variable like `elevation` on lat/lon grids;
        seafloor depth = -elevation where elevation < 0.
      - for each cast position (lat, lon, distance_km), sample the nearest grid
        cell: `ds.elevation.sel(lat=..., lon=..., method="nearest")`.
      - return a DataFrame: distance_km, seafloor_depth_m, interpolated to the
        transect's distance axis.
    Requires `xarray` and `netCDF4` (add to the optional deps when you enable it).
    """
    raise NotImplementedError(
        "Bathymetry reading is a scaffold. Download your bathymetry netCDF "
        "(e.g. GEBCO), install xarray + netCDF4, and implement this using the "
        "guidance in the docstring. Until then, grid_transect_2d still runs; it "
        "simply will not mask below the seafloor."
    )


def apply_bathymetry_mask(grid: dict, seafloor_depth_km: pd.DataFrame) -> dict:
    """Mask grid cells deeper than the seafloor at each distance. Expects
    `seafloor_depth_km` with columns distance_km, seafloor_depth_m."""
    xs = grid["distance_km"]; zs = grid["depth_m"]; vals = grid["values"].copy()
    sf = np.interp(xs, seafloor_depth_km["distance_km"], seafloor_depth_km["seafloor_depth_m"])
    for xi in range(len(xs)):
        vals[zs > sf[xi], xi] = np.nan
    grid = dict(grid); grid["values"] = vals; grid["seafloor_depth_m"] = sf
    return grid


# ---------------------------------------------------------------------------
# Step 5 — plot the section (square pixels)
# ---------------------------------------------------------------------------


def plot_transect_2d(grid: dict, out_path, title: str | None = None,
                     cmap: str = "viridis") -> Path:
    """Plot a 2D section as square pixels (pcolormesh), seafloor masked if
    present, with the model-predicted disclaimer in the caption."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib required. pip install matplotlib") from exc

    xs, zs, vals = grid["distance_km"], grid["depth_m"], grid["values"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mesh = ax.pcolormesh(xs, zs, np.ma.masked_invalid(vals), shading="auto", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label=grid["value_column"])

    if "seafloor_depth_m" in grid:
        ax.fill_between(xs, grid["seafloor_depth_m"], zs.max(), color="#5b4636", zorder=3)

    # mark transect points (stations) with labels on the top axis
    cp = grid.get("cast_positions")
    if cp is not None:
        for _, row in cp.iterrows():
            d = row["distance_km"]
            ax.axvline(d, color="white", linewidth=0.5, alpha=0.6, zorder=2)
            ax.annotate(str(row.get("point_label", "")), xy=(d, 0),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#222",
                        annotation_clip=False)

    ax.invert_yaxis()
    ax.set_xlabel("Distance along transect (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title or f"Predicted {grid['value_column']} section", fontsize=10)
    fig.text(0.01, 0.005,
             "MODEL-PREDICTED section — values between samples are interpolated, "
             "not measured.", fontsize=6.5, color="#a33")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3D (TODO entry point)
# ---------------------------------------------------------------------------


def assemble_3d_volume(*args, **kwargs):  # pragma: no cover - scaffold
    """TODO: 3D carbonate volume (lon × lat × depth) from multiple transects.

    Guidance: a single transect is 2D (distance × depth). A 3D field needs
    several transects (or a grid of casts) so there is information in BOTH
    horizontal directions. Build it only once you have that spatial coverage:
      - predict 1-m profiles at every cast,
      - interpolate onto a regular (lon, lat, depth) grid (start with linear /
        nearest; consider objective mapping with horizontal+vertical length
        scales),
      - mask below the GEBCO seafloor in 3D,
      - export as a CF-compliant netCDF (dims: depth, lat, lon) so it opens in
        Ocean Data View / Panoply, labelled as a predicted field.
    Requires xarray + scipy. This is a genuine research step, not just code.
    """
    raise NotImplementedError(
        "3D volume assembly is a scaffold. It requires multiple transects / a "
        "cast grid and a validated model. See the docstring for the build path."
    )
