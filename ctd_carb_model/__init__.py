"""
ctd_carb_model
==============
Modelling pipeline that predicts carbonate chemistry (pH, TA) from CTD
hydrography, recomputes derived variables (DIC, pCO2, omega) from the predicted
pair, and compares MLR vs ANN with leakage-safe validation.

Reads the bottle-matched master table produced by the ctd_carb_merge tool.

Five parts:
  data       - prepare the master table; label measured vs calculated responses
  models     - MLR and ANN on a common interface; PyCO2SYS recompute
  validation - leave-one-cast/cruise/season-out (leakage-safe)
  compare    - fair head-to-head model comparison with a reasoned winner
  report     - diagnostics plots, report, manifest
"""
from __future__ import annotations

__version__ = "0.1.0"

from .data import prepare, summarise, DataConfig, ModelDataset
from .models import build_model, recompute_carbonate_system, MLRModel, ANNModel
from .compare import compare_models, format_comparison
from .selection import select_predictors, rank_subsets, interpret
from .predict import predict_profiles, write_predicted_profiles

__all__ = ["__version__", "prepare", "summarise", "DataConfig", "ModelDataset",
           "build_model", "recompute_carbonate_system", "MLRModel", "ANNModel",
           "compare_models", "format_comparison", "select_predictors", "rank_subsets", "interpret", "predict_profiles", "write_predicted_profiles"]
