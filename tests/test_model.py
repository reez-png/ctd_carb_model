"""
tests/test_model.py
===================
Tests for the modelling pipeline: data prep, models, validation, comparison,
reporting. ANN/recompute/plot tests skip cleanly if their optional deps are
absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctd_carb_model import data, models, validation, compare, report


def _df(n=40, n_casts=8, seed=1, signal=True):
    rng = np.random.default_rng(seed)
    o = rng.uniform(100, 230, n); d = rng.uniform(4, 85, n)
    s = rng.uniform(35, 35.9, n); t = 29 - d * 0.1 + rng.normal(0, 0.3, n)
    if signal:
        ph = 8.1 - 0.004 * d + 0.0008 * (o - 200) + rng.normal(0, 0.01, n)
        ta = 2280 + (s - 35) * 60 + rng.normal(0, 4, n)
    else:
        ph = rng.uniform(7.7, 8.1, n); ta = rng.uniform(2200, 2400, n)
    return pd.DataFrame({
        "cast_id": [f"P45_06_CTD_{(i % n_casts) + 1:02d}" for i in range(n)],
        "cruise_id": ["P45_06"] * n,
        "ctd_temperature_c": t, "ctd_salinity": s, "ctd_depth_m": d,
        "oxygen_umol_l": o,
        "ph_best": ph, "ta_best_umolkg": ta,
        "dic_best_umol_kg": rng.uniform(1950, 2150, n),
        "pco2_best_uatm": rng.uniform(350, 800, n),
        "omega_ar": rng.uniform(1.1, 3.2, n), "omega_ca": rng.uniform(1.8, 4.5, n),
        "analysis_audit_status": ["PASS"] * (n - 2) + ["FAIL"] * 2,
        "flag_no_ctd_match": [False] * n,
    })


# -- Part 1: data ----------------------------------------------------------

def test_prepare_resolves_real_column_names():
    ds = data.prepare(_df())
    assert "ctd_oxygen" in ds.predictors          # resolved from oxygen_umol_l
    assert "ph_total" in ds.responses             # resolved from ph_best
    assert "total_alkalinity_umol_kg" in ds.responses

def test_prepare_labels_measured_vs_calculated():
    ds = data.prepare(_df())
    assert ds.response_status["ph_total"] == "measured"
    assert ds.response_status["dic_umol_kg"] == "calculated"
    assert ds.response_status["omega_aragonite"] == "calculated"

def test_prepare_drops_fail_rows():
    ds = data.prepare(_df(n=40))
    assert ds.n_rows == 38   # 2 FAIL dropped


# -- Part 2: models --------------------------------------------------------

def test_mlr_fits_and_predicts():
    ds = data.prepare(_df())
    m = models.build_model("MLR", ds.predictors).fit(ds.frame, "ph_total")
    p = m.predict(ds.frame.head(3))
    assert len(p) == 3 and np.all(np.isfinite(p))
    assert "intercept" in m.coefficients

def test_common_interface_both_models():
    sk = pytest.importorskip("sklearn")
    ds = data.prepare(_df())
    for name in ["MLR", "ANN"]:
        m = models.build_model(name, ds.predictors).fit(ds.frame, "ph_total")
        assert len(m.predict(ds.frame.head(2))) == 2

def test_recompute_carbonate_system():
    pytest.importorskip("PyCO2SYS")
    ds = data.prepare(_df())
    ds.frame["ph_pred"] = ds.frame["ph_total"]
    ds.frame["ta_pred"] = ds.frame["total_alkalinity_umol_kg"]
    rec = models.recompute_carbonate_system(ds.frame, "ph_pred", "ta_pred")
    assert "dic_umol_kg_recomputed" in rec.columns
    assert (rec["omega_aragonite_recomputed"] > 0).all()


# -- Part 3: validation ----------------------------------------------------

def test_leave_one_cast_out_no_leakage():
    ds = data.prepare(_df())
    cv = validation.cross_validate(ds.frame, lambda: models.build_model("MLR", ds.predictors),
                                   "ph_total", "leave_one_cast_out")
    assert cv["ok"] and cv["n_folds"] >= 2

def test_mlr_out_of_group_recovers_signal():
    ds = data.prepare(_df(signal=True))
    cv = validation.cross_validate(ds.frame, lambda: models.build_model("MLR", ds.predictors),
                                   "ph_total", "leave_one_cast_out")
    assert cv["pooled"]["r2"] > 0.5   # real signal -> good out-of-group fit

def test_validation_refuses_single_cruise():
    ds = data.prepare(_df())
    cv = validation.cross_validate(ds.frame, lambda: models.build_model("MLR", ds.predictors),
                                   "ph_total", "leave_one_cruise_out")
    assert not cv["ok"] and "one cruise" in cv["note"]


# -- Part 4: comparison ----------------------------------------------------

def test_compare_picks_mlr_when_signal():
    pytest.importorskip("sklearn")
    ds = data.prepare(_df(signal=True))
    res = compare.compare_models(ds.frame, ds.predictors, ["ph_total"], ["MLR", "ANN"])
    w = res["winners"].set_index("response").loc["ph_total", "winner"]
    assert w == "MLR"   # MLR should win / be preferred on small data

def test_compare_declares_no_winner_when_noise():
    ds = data.prepare(_df(signal=False, seed=5))
    res = compare.compare_models(ds.frame, ds.predictors, ["ph_total"], ["MLR"])
    # pure noise -> R2 <= 0 -> no usable model
    w = res["winners"].set_index("response").loc["ph_total", "winner"]
    assert w == "none"


# -- Part 5: report --------------------------------------------------------

def test_generate_all_writes_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    ds = data.prepare(_df())
    res = compare.compare_models(ds.frame, ds.predictors, ["ph_total"], ["MLR"])
    written = report.generate_all(res, tmp_path / "out", ds.response_status)
    assert any("pred_vs_actual" in p for p in written["plots"])
    report.write_report(res, tmp_path / "out" / "report.md", data.summarise(ds), ds.response_status)
    report.write_manifest(res, tmp_path / "out" / "manifest.json")
    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


# -- AICc predictor selection ---------------------------------------------

from ctd_carb_model import selection


def _df_signal_no_salinity(n=38, seed=3):
    rng = np.random.default_rng(seed)
    o = rng.uniform(100, 230, n); d = rng.uniform(4, 85, n)
    s = rng.uniform(35, 35.9, n); t = 29 - d * 0.1 + rng.normal(0, 0.3, n)
    ph = 8.1 - 0.004 * d + 0.0008 * (o - 200) + rng.normal(0, 0.01, n)  # no salinity
    return pd.DataFrame({
        "cast_id": [f"P45_06_CTD_{(i % 8) + 1:02d}" for i in range(n)],
        "cruise_id": ["P45_06"] * n,
        "ctd_temperature_c": t, "ctd_salinity": s, "ctd_depth_m": d, "oxygen_umol_l": o,
        "ph_best": ph, "ta_best_umolkg": 2280 + (s - 35) * 60 + rng.normal(0, 4, n),
        "analysis_audit_status": ["PASS"] * n, "flag_no_ctd_match": [False] * n,
    })


def test_aicc_ranking_has_criteria_and_weights():
    ds = data.prepare(_df_signal_no_salinity())
    ranking = selection.rank_subsets(ds.frame, "ph_total", ds.predictors, "aicc")
    assert {"aic", "aicc", "bic", "delta_aicc", "weight_aicc"}.issubset(ranking.columns)
    assert ranking.iloc[0]["delta_aicc"] == 0.0
    assert abs(ranking["weight_aicc"].sum() - 1.0) < 1e-2


def test_aicc_drops_irrelevant_predictor():
    ds = data.prepare(_df_signal_no_salinity())
    best, ranking = selection.select_predictors(ds.frame, "ph_total", ds.predictors, "aicc")
    # depth and oxygen are the true drivers; both should be selected
    assert "ctd_depth_m" in best and "ctd_oxygen" in best
    # the full 4-predictor model should not be the single best
    assert len(best) < len(ds.predictors)


def test_aicc_subsets_fit_on_identical_rows():
    ds = data.prepare(_df_signal_no_salinity())
    ranking = selection.rank_subsets(ds.frame, "ph_total", ds.predictors, "aicc")
    assert ranking["n"].nunique() == 1


def test_compare_with_selection_runs():
    pytest.importorskip("sklearn")
    ds = data.prepare(_df_signal_no_salinity())
    result = compare.compare_models(ds.frame, ds.predictors, ["ph_total"],
                                    ["MLR", "ANN"], select_by="aicc")
    assert "selection" in result
    assert "ph_total" in result["selection"]
    assert result["selection"]["ph_total"]["predictors"]  # non-empty chosen set


def test_sparse_predictor_dropped_no_crash():
    # oxygen entirely missing must not crash compare; it should be dropped
    rng = np.random.default_rng(0); n = 38
    d = rng.uniform(4, 85, n); s = rng.uniform(35, 35.9, n)
    df = pd.DataFrame({
        "cast_id": [f"P45_06_CTD_{(i % 8) + 1:02d}" for i in range(n)],
        "cruise_id": ["P45_06"] * n,
        "ctd_temperature_c": 29 - d * 0.1, "ctd_salinity": s, "ctd_depth_m": d,
        "oxygen_umol_l": [np.nan] * n,                       # entirely missing
        "ph_best": 8.1 - 0.004 * d + rng.normal(0, 0.01, n),
        "ta_best_umolkg": 2280 + (s - 35) * 60,
        "analysis_audit_status": ["PASS"] * n, "flag_no_ctd_match": [False] * n,
    })
    ds = data.prepare(df)
    assert "ctd_oxygen" not in ds.predictors        # dropped for low coverage
    res = compare.compare_models(ds.frame, ds.predictors, ["ph_total"], ["MLR"], select_by="aicc")
    # must not crash and must return a winners table
    assert "winner" in res["winners"].columns


def test_partial_nan_predictor_no_crash():
    pytest.importorskip("sklearn")
    # oxygen present in ~55% of rows: ANN must not receive NaN inputs
    rng = np.random.default_rng(9); n = 38
    d = rng.uniform(4, 85, n); s = rng.uniform(35, 35.9, n)
    o = rng.uniform(100, 230, n).astype(float); o[rng.random(n) < 0.45] = np.nan
    df = pd.DataFrame({
        "cast_id": [f"P45_06_CTD_{(i % 8) + 1:02d}" for i in range(n)],
        "cruise_id": ["P45_06"] * n,
        "ctd_temperature_c": 29 - d * 0.1, "ctd_salinity": s, "ctd_depth_m": d,
        "oxygen_umol_l": o,
        "ph_best": 8.1 - 0.004 * d + rng.normal(0, 0.01, n),
        "ta_best_umolkg": 2280 + (s - 35) * 60,
        "analysis_audit_status": ["PASS"] * n, "flag_no_ctd_match": [False] * n,
    })
    ds = data.prepare(df)
    # must not raise even with both models and AICc selection
    res = compare.compare_models(ds.frame, ds.predictors, ["ph_total", "total_alkalinity_umol_kg"],
                                 ["MLR", "ANN"], select_by="aicc")
    assert "winner" in res["winners"].columns


def test_predict_profiles_flags_extrapolation():
    import numpy as np, pandas as pd
    from ctd_carb_model import predict
    rng = np.random.default_rng(0)
    brows = []
    for i in range(38):
        d = rng.uniform(4, 85); s = rng.uniform(35, 35.9); o = 200 - d * 1.4
        brows.append(dict(cast_id=f"P45_06_CTD_{(i % 8) + 1:02d}", cruise_id="P45_06",
                          ctd_temperature_c=29 - d * 0.1, ctd_salinity=s, ctd_depth_m=d,
                          ctd_oxygen=o,
                          ph_best=8.1 - 0.004 * d + rng.normal(0, 0.01),
                          ta_best_umolkg=2280 + (s - 35) * 60 + rng.normal(0, 4),
                          analysis_audit_status="PASS", flag_no_ctd_match=False))
    bottle = pd.DataFrame(brows)
    prows = []
    for cruise in ["P45_05", "P45_06"]:
        for ci in range(2):
            for z in np.arange(0, 41):
                prows.append(dict(cast_id=f"{cruise}_CTD_{ci+1:02d}", cruise_id=cruise,
                                  station_name=f"J{ci+1}", latitude=5.5 + ci * 0.05,
                                  longitude=0.5 + ci * 0.05, temperature_degC=29 - z * 0.1,
                                  salinity_psu=35.2 + z * 0.01, oxygen_umol_kg=200 - z * 1.4,
                                  depth_m=float(z)))
    profile = pd.DataFrame(prows)
    pred, info = predict.predict_profiles(bottle, profile, select_by="aicc")
    # predicted columns exist
    assert "ph_total_predicted" in pred.columns
    assert "total_alkalinity_umol_kg_predicted" in pred.columns
    # derived recomputed from predicted pair
    assert "dic_umol_kg_predicted" in pred.columns
    # extrapolation flag: P45_05 True, P45_06 False
    assert pred[pred.cruise_id == "P45_05"]["is_extrapolation"].all()
    assert (~pred[pred.cruise_id == "P45_06"]["is_extrapolation"]).all()
