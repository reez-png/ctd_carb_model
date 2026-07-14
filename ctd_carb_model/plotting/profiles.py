"""
ctd_carb_analysis.profiles
==========================
Observed vertical profile figures: aligned panels with CTD variables drawn as
continuous lines and discrete carbonate variables drawn as markers at their
sampled depths. Depth increases downward on every panel.

This is "Product B" in the project design. The functions enforce the
measured-vs-predicted visual rule: CTD = solid line, discrete measured =
filled circle, calculated-at-sample = distinct filled marker, predicted =
dashed line with an uncertainty ribbon (only used by the modelling branch).

A discrete carbonate series may be connected with a thin dashed line purely as
a visual aid; the function labels it explicitly as a visual connection between
discrete observations, never as a measured profile.

matplotlib is imported lazily so importing this module never requires a display
or the plotting stack until you actually draw something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class PanelSpec:
    """One panel of a profile figure."""

    column: str                     # column in the data
    label: str                      # axis label
    kind: str = "ctd_line"          # ctd_line | discrete_marker | predicted_band
    connect_discrete: bool = True   # thin visual line between discrete points


# Default panel layout matching the project design (Temperature, Salinity,
# Oxygen as CTD lines; pH_T, TA, Omega_arag as discrete markers).
DEFAULT_PANELS = [
    PanelSpec("ctd_temperature_c", "Temperature (°C)", "ctd_line"),
    PanelSpec("ctd_salinity", "Salinity (PSU)", "ctd_line"),
    PanelSpec("ctd_oxygen", "Dissolved O₂ (µmol kg⁻¹)", "ctd_line"),
    PanelSpec("ph_total", "pH (total scale)", "discrete_marker"),
    PanelSpec("total_alkalinity_umol_kg", "TA (µmol kg⁻¹)", "discrete_marker"),
    PanelSpec("omega_aragonite", "Ω aragonite", "discrete_marker"),
]


@dataclass
class ProfileFigureConfig:
    depth_column_carbonate: str = "depth_round_m"
    depth_column_ctd: str = "depth_m"
    cast_column: str = "cast_id"
    station_column: str = "station_name"
    panels: list[PanelSpec] = field(default_factory=lambda: list(DEFAULT_PANELS))
    figure_height: float = 6.0
    panel_width: float = 2.4


def plot_station_profile(
    matched: pd.DataFrame,
    ctd_profile: pd.DataFrame | None,
    cast_id: str,
    out_path: str | Path,
    config: ProfileFigureConfig | None = None,
    title: str | None = None,
):
    """Draw one station/cast profile figure and save it.

    Parameters
    ----------
    matched : the bottle-matched table (output of the merge), filtered or not.
        Discrete carbonate columns are read from here, at depth
        `config.depth_column_carbonate`.
    ctd_profile : the continuous CTD profile for this cast (depth + CTD vars).
        If None, only discrete markers are drawn (CTD line panels are left
        empty with a note). Pass the cast's rows from the gathered CTD frame.
    cast_id : which cast this figure is for (used to filter and title).
    out_path : where to save the PNG.

    Returns the saved path. Raises a clear error if matplotlib is unavailable.
    """
    config = config or ProfileFigureConfig()
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for profile figures. "
            "Install it with: pip install matplotlib"
        ) from exc

    sub = matched[matched[config.cast_column].astype(str) == str(cast_id)].copy()
    panels = config.panels
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(config.panel_width * n, config.figure_height), sharey=True
    )
    if n == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel.kind == "ctd_line" and ctd_profile is not None and panel.column in ctd_profile:
            cp = ctd_profile.sort_values(config.depth_column_ctd)
            ax.plot(
                pd.to_numeric(cp[panel.column], errors="coerce"),
                pd.to_numeric(cp[config.depth_column_ctd], errors="coerce"),
                "-", color="#1f4e79", linewidth=1.6,
            )
        elif panel.kind == "discrete_marker" and panel.column in sub:
            d = sub.sort_values(config.depth_column_carbonate)
            x = pd.to_numeric(d[panel.column], errors="coerce")
            y = pd.to_numeric(d[config.depth_column_carbonate], errors="coerce")
            if panel.connect_discrete and x.notna().sum() > 1:
                # thin dashed visual aid ONLY — labelled as such
                ax.plot(x, y, "--", color="#888888", linewidth=0.8, zorder=1)
            ax.plot(x, y, "o", color="#b1422d", markersize=6, zorder=2)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#999999", fontsize=8)

        ax.set_xlabel(panel.label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()  # depth increases downward
    fig.suptitle(title or f"Cast {cast_id}", fontsize=11)

    # Caption enforcing the measured-vs-aid distinction.
    fig.text(
        0.01, 0.005,
        "Solid line: CTD measured profile.  Filled circles: discrete carbonate "
        "measurements/calculations.  Dashed grey: visual aid between discrete "
        "points only (not a measured profile).",
        fontsize=6.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all_station_profiles(
    matched: pd.DataFrame,
    ctd: pd.DataFrame | None,
    out_dir: str | Path,
    config: ProfileFigureConfig | None = None,
) -> list[Path]:
    """Draw one figure per cast present in the matched table."""
    config = config or ProfileFigureConfig()
    out_dir = Path(out_dir)
    saved: list[Path] = []
    for cast_id in matched[config.cast_column].dropna().astype(str).unique():
        cast_ctd = None
        if ctd is not None and config.cast_column in ctd.columns:
            cast_ctd = ctd[ctd[config.cast_column].astype(str) == cast_id]
        path = plot_station_profile(
            matched, cast_ctd, cast_id,
            out_dir / f"profile_{cast_id}.png", config,
        )
        saved.append(path)
    return saved
