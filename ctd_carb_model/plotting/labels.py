"""
ctd_carb_analysis.labels
========================
The canonical vocabulary that keeps measured, calculated, and predicted values
distinct. Everything else in the package references these so the distinction is
applied consistently in tables, figures, and metadata.

The scientific rule (from the project design): a value's *status* must travel
with it. A discrete laboratory pH is not the same kind of object as a pH
calculated by PyCO2SYS from a measured pair, which is not the same as a pH
predicted by a regional model at an unsampled depth. They share a unit but not a
provenance, and they must never be plotted, tabulated, or described as if they
were interchangeable.
"""

from __future__ import annotations

from enum import Enum


class MeasurementStatus(str, Enum):
    """How a given value came to exist."""

    MEASURED = "measured"          # direct laboratory / sensor measurement
    CALCULATED = "calculated"      # derived from measured inputs (e.g. PyCO2SYS)
    PREDICTED = "predicted"        # estimated by a model at an unsampled point
    CTD_OBSERVED = "ctd_observed"  # measured by the CTD at a 1-m level


# Which carbonate variables are typically measured vs calculated in this
# project's workflow (measured pH + measured TA -> calculate the rest).
CARBONATE_STATUS_DEFAULTS = {
    "ph_total": MeasurementStatus.MEASURED,
    "total_alkalinity_umol_kg": MeasurementStatus.MEASURED,
    "dic_umol_kg": MeasurementStatus.CALCULATED,
    "pco2_uatm": MeasurementStatus.CALCULATED,
    "omega_aragonite": MeasurementStatus.CALCULATED,
    "omega_calcite": MeasurementStatus.CALCULATED,
}


# The three product kinds the project produces, kept in separate output trees.
PRODUCT_KINDS = {
    "observed_matched": (
        "One row per discrete water sample, with the matched CTD conditions "
        "attached. Measured carbonate values plus their directly calculated "
        "carbonate products. Defensible immediately."
    ),
    "ctd_profile_with_markers": (
        "The full CTD water column with a flag marking the depths where "
        "discrete carbonate samples were collected. For visualising sampling "
        "coverage against water-column structure."
    ),
    "predicted_profile": (
        "Model-predicted continuous carbonate profile at CTD depth bins. "
        "Created only after regional model calibration and independent "
        "validation. Must always be labelled as model-predicted, never as a "
        "measured 1-m carbonate profile."
    ),
}


# Standard legend wording the figures and reports reuse verbatim.
LEGEND_WORDING = {
    MeasurementStatus.CTD_OBSERVED: "CTD measured profile (continuous line)",
    MeasurementStatus.MEASURED: "Discrete carbonate measurement (filled circle)",
    MeasurementStatus.CALCULATED: "Calculated from measured pair (filled marker)",
    MeasurementStatus.PREDICTED: "Model-predicted profile (dashed line + uncertainty)",
}

# The exact sentence a predicted product must carry.
PREDICTED_PROFILE_DISCLAIMER = (
    "Model-predicted carbonate profile generated from CTD hydrographic "
    "predictors and validated discrete carbonate observations. This is NOT a "
    "measured 1-m carbonate profile."
)
