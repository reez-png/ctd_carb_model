"""
ctd_carb_model.selection
========================
Data-driven predictor selection by AICc (exhaustive subset ranking).

Why AICc (not plain AIC): AICc is AIC with a small-sample correction. With ~tens
of samples relative to a handful of predictors, that correction matters; AICc
guards against picking an over-large model. As n grows, AICc -> AIC.

How it is used (and the honest division of labour):
  - selection ranks predictor subsets on IN-SAMPLE fit penalised by complexity.
  - it is run ONCE on the full data to choose a subset (stable at small n).
  - the chosen subset is then validated OUT-OF-GROUP by Part 3 (leave-one-cast-
    out). Selection answers "which predictors?"; validation answers "does the
    chosen model actually predict?". Report both.

Comparable models: every subset is fit on the SAME complete-case rows (the
intersection where the response and ALL candidate predictors are present), so
the AICc values are comparable across subsets.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _ols_ss_res(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return float(np.sum(resid ** 2)), beta


def information_criteria(n: int, k_params: int, ss_res: float) -> dict:
    """AIC, AICc, BIC for a Gaussian OLS fit.

    k_params = number of estimated parameters INCLUDING intercept and the
    residual variance, i.e. (n_predictors + 2).
    AICc is undefined when (n - k_params - 1) <= 0; we return inf there so such
    over-parameterised subsets sort to the bottom.
    """
    if n <= 0 or ss_res <= 0:
        return {"aic": np.nan, "aicc": np.nan, "bic": np.nan, "log_lik": np.nan}
    sigma2 = ss_res / n
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * k_params - 2 * log_lik
    denom = n - k_params - 1
    aicc = aic + (2 * k_params * (k_params + 1) / denom) if denom > 0 else float("inf")
    bic = k_params * np.log(n) - 2 * log_lik
    return {"aic": float(aic), "aicc": float(aicc), "bic": float(bic),
            "log_lik": float(log_lik)}


def rank_subsets(
    df: pd.DataFrame,
    response: str,
    candidate_predictors: list[str],
    criterion: str = "aicc",
    min_predictors: int = 1,
) -> pd.DataFrame:
    """Rank every non-empty subset of candidate_predictors by `criterion`.

    Returns a tidy table sorted best-first with R2, adjusted R2, RMSE, AIC,
    AICc, BIC, the delta on the chosen criterion, and Akaike weights.

    With c candidates there are 2^c - 1 subsets (c=4 -> 15, c=6 -> 63): trivial.
    Interpretation (Burnham & Anderson): delta < 2 ~ equivalent support;
    4-7 considerably less; > 10 essentially none.
    """
    crit = criterion.lower()
    if crit not in {"aic", "aicc", "bic"}:
        raise ValueError("criterion must be 'aic', 'aicc', or 'bic'")
    present = [p for p in candidate_predictors if p in df.columns]
    if response not in df.columns or len(present) < min_predictors:
        raise KeyError(f"Need response {response!r} and >= {min_predictors} "
                       f"present predictor(s); have {present}.")

    # Drop candidate predictors with too little coverage. Otherwise requiring
    # complete cases across ALL candidates lets one near-empty column (e.g. a
    # sparse oxygen channel) collapse the usable row count to near zero.
    min_coverage = 0.5  # a candidate must be present in >= 50% of rows with a response
    has_resp = df[df[response].notna()] if response in df.columns else df
    kept = []
    for p in present:
        cov = pd.to_numeric(has_resp[p], errors="coerce").notna().mean() if len(has_resp) else 0
        if cov >= min_coverage:
            kept.append(p)
    if len(kept) < min_predictors:
        # fall back to the best-covered single predictor so selection still runs
        covs = {p: pd.to_numeric(has_resp[p], errors="coerce").notna().mean() for p in present}
        kept = [max(covs, key=covs.get)] if covs else present
    present = kept

    # fix the row set once so all subsets are comparable
    complete = df[[response] + present].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(complete)
    y = complete[response].to_numpy(float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    rows = []
    for k in range(min_predictors, len(present) + 1):
        for subset in combinations(present, k):
            k_params = k + 2  # predictors + intercept + variance
            if n - k_params - 1 <= 0:
                continue
            X = complete[list(subset)].to_numpy(float)
            ss_res, _ = _ols_ss_res(X, y)
            ic = information_criteria(n, k_params, ss_res)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            adj_r2 = (1 - (1 - r2) * (n - 1) / (n - k - 1)) if (n - k - 1) > 0 else np.nan
            rows.append({
                "response": response,
                "predictors": ", ".join(subset),
                "n_predictors": k, "n": n,
                "r2": r2, "adj_r2": adj_r2,
                "rmse": float(np.sqrt(ss_res / n)),
                "aic": ic["aic"], "aicc": ic["aicc"], "bic": ic["bic"],
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(crit).reset_index(drop=True)
    best = out[crit].min()
    out[f"delta_{crit}"] = out[crit] - best
    w = np.exp(-0.5 * out[f"delta_{crit}"].clip(upper=700))
    out[f"weight_{crit}"] = (w / w.sum()).round(4)
    return out


def select_predictors(
    df: pd.DataFrame,
    response: str,
    candidate_predictors: list[str],
    criterion: str = "aicc",
) -> tuple[list[str], pd.DataFrame]:
    """Return (best_subset, full_ranking_table) for one response."""
    ranking = rank_subsets(df, response, candidate_predictors, criterion)
    if ranking.empty:
        raise ValueError(f"No fittable subset for {response!r} (too little data).")
    best = ranking.iloc[0]["predictors"].split(", ")
    return best, ranking


def interpret(ranking: pd.DataFrame, criterion: str = "aicc") -> str:
    """A short, honest reading of a ranking table."""
    if ranking.empty:
        return "No subsets could be fit."
    crit = criterion.lower()
    top = ranking.iloc[0]
    near = ranking[ranking[f"delta_{crit}"] < 2]
    msg = [f"Best by {crit.upper()}: [{top['predictors']}] "
           f"(R²={top['r2']:.3f}, weight={top[f'weight_{crit}']:.2f})."]
    if len(near) > 1:
        msg.append(f"{len(near)} subsets within Δ{crit.upper()}<2 — the data do "
                   "not strongly single out one; the top model is not decisively "
                   "best. Consider reporting the candidate set, not just the winner.")
    else:
        msg.append(f"The top model is clearly preferred (next Δ{crit.upper()}="
                   f"{ranking.iloc[1][f'delta_{crit}']:.1f}).")
    return " ".join(msg)
