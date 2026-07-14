"""
ctd_carb_model.compare
======================
PART 4 of the modelling pipeline: the head-to-head comparison.

Fairness is the whole point. Every model is run through the SAME validation
(same rows, same cast folds, same metrics, same response) via Part 3's
cross_validate, so differences in the table reflect the models, not the setup.

The comparison is performed on the MEASURED responses (pH, TA) — the variables
we actually fit. The derived variables (DIC, pCO2, omega) are obtained later by
recomputing from the winning model's predicted pH+TA, so they are not separate
contests.

Outputs:
  - compare_models(): a tidy table, one row per (response, model), with honest
    out-of-group metrics and a per-response winner + reason.
  - choose_winner(): the decision rule, stated explicitly so it is auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import validation
from .models import build_model


def compare_models(
    df: pd.DataFrame,
    predictors: list[str],
    responses: list[str],
    model_names: list[str] | None = None,
    split: str = "leave_one_cast_out",
    model_kwargs: dict | None = None,
    select_by: str | None = None,
) -> dict:
    """Run every model through the same validation for each measured response.

    If `select_by` is given ('aicc', 'aic', or 'bic'), predictors are chosen per
    response by exhaustive subset ranking on the FULL data first, then both
    models are validated on that chosen subset. The chosen subset and the full
    ranking are returned under 'selection'. If None, all `predictors` are used.

    Returns a dict with:
      - 'table': tidy DataFrame (response, model, n, r2, rmse, mae, bias, n_folds, ok)
      - 'winners': DataFrame (response, winner, reason)
      - 'selection': {response: {'predictors': [...], 'ranking': DataFrame, 'note': str}}
      - 'split': the validation split used
    """
    model_names = model_names or ["MLR", "ANN"]
    model_kwargs = model_kwargs or {}

    rows = []
    cv_store: dict[tuple, dict] = {}
    selection_store: dict[str, dict] = {}
    for response in responses:
        # choose predictors for this response
        resp_predictors = predictors
        if select_by:
            from . import selection as _sel
            try:
                chosen, ranking = _sel.select_predictors(df, response, predictors, select_by)
                resp_predictors = chosen
                selection_store[response] = {
                    "predictors": chosen,
                    "ranking": ranking,
                    "note": _sel.interpret(ranking, select_by),
                }
            except (ValueError, KeyError) as exc:
                selection_store[response] = {"predictors": predictors,
                                             "ranking": None, "note": str(exc)}
        for name in model_names:
            kw = model_kwargs.get(name, {})
            factory = lambda name=name, kw=kw, rp=resp_predictors: build_model(name, rp, **kw)
            cv = validation.cross_validate(df, factory, response, split)
            cv_store[(response, name)] = cv
            if not cv.get("ok"):
                rows.append({"response": response, "model": name, "ok": False,
                             "note": cv.get("note", "")})
                continue
            p = cv["pooled"]
            rows.append({
                "response": response, "model": name, "ok": True,
                "predictors": ", ".join(resp_predictors),
                "n": p["n"], "r2": p["r2"], "rmse": p["rmse"],
                "mae": p["mae"], "bias": p["bias"], "n_folds": cv["n_folds"],
            })

    table = pd.DataFrame(rows)
    winners = _winners(table)
    return {"table": table, "winners": winners, "split": split, "cv": cv_store,
            "selection": selection_store}


def choose_winner(group: pd.DataFrame) -> tuple[str, str]:
    """Decision rule for one response, stated explicitly so it can be audited.

    Rule, in order:
      1. Consider only models that validated (ok and finite R2).
      2. If none, there is no winner — the data cannot support a model here.
      3. If the best out-of-group R2 is <= 0, declare NO usable model (a model
         no better than predicting the mean is not a model).
      4. Otherwise the winner is the highest out-of-group R2. If a simpler model
         (MLR) is within a small margin of a complex one (ANN), prefer the
         simpler — parsimony, and it generalises more safely on small data.
    """
    g = group[group.get("ok", False) == True].copy()
    if g.empty or "r2" not in g.columns:
        return ("none", "no model validated (insufficient complete-case data).")
    g = g[np.isfinite(g["r2"])]
    if g.empty:
        return ("none", "no model validated (insufficient data).")

    best = g.loc[g["r2"].idxmax()]
    if best["r2"] <= 0:
        return ("none",
                f"best out-of-group R2={best['r2']:.2f} <= 0 — no model beats "
                "predicting the mean; collect more data before modelling.")

    # parsimony tie-break: prefer MLR if it is within 0.02 R2 of the best
    if "MLR" in set(g["model"]):
        mlr_r2 = float(g.loc[g["model"] == "MLR", "r2"].iloc[0])
        if best["model"] != "MLR" and (best["r2"] - mlr_r2) < 0.02:
            return ("MLR",
                    f"MLR (R2={mlr_r2:.3f}) within 0.02 of {best['model']} "
                    f"(R2={best['r2']:.3f}); prefer the simpler, safer model.")
    return (best["model"],
            f"highest out-of-group R2={best['r2']:.3f} (RMSE={best['rmse']:.4g}).")


def _winners(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "response" not in table.columns:
        return pd.DataFrame(columns=["response", "winner", "reason"])
    out = []
    for response, group in table.groupby("response"):
        winner, reason = choose_winner(group)
        out.append({"response": response, "winner": winner, "reason": reason})
    return pd.DataFrame(out)


def format_comparison(result: dict) -> str:
    """Human-readable summary for the console / report."""
    table = result["table"]
    winners = result["winners"]
    lines = [f"Model comparison (validation: {result['split']})", ""]
    selection = result.get("selection") or {}
    if selection:
        lines.append("Predictor selection (AICc, on full data):")
        for resp, info in selection.items():
            lines.append(f"  {resp}: [{', '.join(info['predictors'])}]")
            if info.get("note"):
                lines.append(f"      {info['note']}")
        lines.append("")
    if not table.empty:
        cols = [c for c in ["response", "model", "n", "r2", "rmse", "mae", "bias", "n_folds"]
                if c in table.columns]
        show = table[table.get("ok", True) == True][cols] if "ok" in table.columns else table[cols]
        if not show.empty:
            lines.append(show.round(4).to_string(index=False))
    lines.append("")
    lines.append("Winners:")
    for _, w in winners.iterrows():
        lines.append(f"  {w['response']}: {w['winner']}  — {w['reason']}")
    return "\n".join(lines)
