"""
ctd_carb_model.models
=====================
PART 2 of the modelling pipeline: the models themselves.

Design principle: MLR and ANN implement the SAME small interface, so that
validation (Part 3) and comparison (Part 4) can treat them identically and the
comparison is genuinely fair — same inputs, same outputs, same folds, same
metrics. The only thing that differs between them is what happens inside fit().

We only fit models for the MEASURED variables (pH, TA). The CALCULATED variables
(DIC, pCO2, omega) are obtained by recomputing them from the predicted pH+TA with
PyCO2SYS — the same way the observed values were produced — so the derived
predictions stay carbonate-system-consistent. That recompute lives in
recompute_carbonate_system() at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


class CarbModel:
    """Base class. A model maps CTD predictors -> one carbonate response.

    Subclasses implement _fit and _predict on plain numpy arrays. The base class
    handles column selection, NaN dropping, and standardisation so every model
    sees the same clean inputs.
    """

    name = "base"
    needs_scaling = False

    def __init__(self, predictors: list[str]):
        self.predictors = predictors
        self._fitted = False
        self._mu = None
        self._sigma = None

    # -- public API used by validation/comparison --------------------------

    def fit(self, df: pd.DataFrame, response: str) -> "CarbModel":
        X, y = self._xy(df, response)
        if self.needs_scaling:
            self._mu = X.mean(axis=0)
            self._sigma = X.std(axis=0)
            self._sigma[self._sigma == 0] = 1.0
            X = (X - self._mu) / self._sigma
        self._fit(X, y)
        self._fitted = True
        self._response = response
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        X = df[self.predictors].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if self.needs_scaling:
            X = (X - self._mu) / self._sigma
        return self._predict(X)

    # -- helpers -----------------------------------------------------------

    def _xy(self, df: pd.DataFrame, response: str):
        sub = df[self.predictors + [response]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) <= len(self.predictors) + 1:
            raise ValueError(
                f"Not enough complete rows ({len(sub)}) to fit {response} on "
                f"{len(self.predictors)} predictors."
            )
        return (sub[self.predictors].to_numpy(float), sub[response].to_numpy(float))

    # -- to be implemented by subclasses -----------------------------------

    def _fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Model 1 — Multiple Linear Regression (OLS)
# ---------------------------------------------------------------------------


class MLRModel(CarbModel):
    """Ordinary least squares. Interpretable, fast, the right baseline. Stores
    coefficients so you can report which predictor matters and in what direction."""

    name = "MLR"
    needs_scaling = False

    def _fit(self, X, y):
        Xd = np.column_stack([np.ones(len(X)), X])
        self.beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)

    def _predict(self, X):
        Xd = np.column_stack([np.ones(len(X)), X])
        return Xd @ self.beta

    @property
    def coefficients(self) -> dict:
        return dict(zip(["intercept"] + self.predictors, self.beta.tolist()))


# ---------------------------------------------------------------------------
# Model 2 — Artificial Neural Network (small MLP)
# ---------------------------------------------------------------------------


class ANNModel(CarbModel):
    """A small feed-forward network. Can capture nonlinear relationships MLR
    misses — but on a small dataset it can also memorise, which is exactly why
    the comparison uses leakage-safe validation rather than in-sample fit.

    Uses scikit-learn's MLPRegressor if available. Deliberately small and
    regularised (modest hidden layer, L2 alpha) because the dataset is small.
    Inputs are standardised (needs_scaling = True) — neural nets require it.
    """

    name = "ANN"
    needs_scaling = True

    def __init__(self, predictors, hidden_layer_sizes=(8,), alpha=1.0,
                 max_iter=2000, random_state=0):
        super().__init__(predictors)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha          # L2 regularisation — higher = simpler, safer on small n
        self.max_iter = max_iter
        self.random_state = random_state

    def _fit(self, X, y):
        try:
            from sklearn.neural_network import MLPRegressor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ANN needs scikit-learn. Install with: pip install scikit-learn"
            ) from exc
        self._net = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._net.fit(X, y)

    def _predict(self, X):
        return self._net.predict(X)


# Registry so the rest of the pipeline can build models by name.
MODEL_REGISTRY = {"MLR": MLRModel, "ANN": ANNModel}


def build_model(name: str, predictors: list[str], **kwargs) -> CarbModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](predictors, **kwargs)


# ---------------------------------------------------------------------------
# Carbonate-system recompute (the defensible path for derived variables)
# ---------------------------------------------------------------------------


# Map the merged data's provenance values to PyCO2SYS option codes where we can;
# fall back to common defaults and RECORD what was used.
def recompute_carbonate_system(
    df: pd.DataFrame,
    ph_column: str,
    ta_column: str,
    temperature_column: str = "ctd_temperature_c",
    salinity_column: str = "ctd_salinity",
    pressure_column: str | None = "ctd_pressure_dbar",
    opt_pH_scale: int = 1,        # 1 = total scale (your data is total scale)
    opt_k_carbonic: int = 10,     # 10 = Lueker et al. 2000 (a common shelf choice)
    opt_k_bisulfate: int = 1,
    opt_total_borate: int = 1,
) -> pd.DataFrame:
    """Given pH and TA columns (measured OR predicted), recompute DIC, pCO2,
    and the omegas with PyCO2SYS at in-situ T/S/P. Returns a frame of the
    recomputed variables, suffixed so you know they are derived.

    Use the SAME constants your observed carbonate values used (read them from
    the merged data's carbonate_constants / carbonate_ph_scale provenance and
    pass them in) so predicted-vs-observed comparisons on derived variables are
    like-for-like.
    """
    try:
        import PyCO2SYS as pyco2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Recompute needs PyCO2SYS. pip install PyCO2SYS") from exc

    ph = pd.to_numeric(df[ph_column], errors="coerce").to_numpy(float)
    ta = pd.to_numeric(df[ta_column], errors="coerce").to_numpy(float)
    temp = pd.to_numeric(df[temperature_column], errors="coerce").to_numpy(float)
    sal = pd.to_numeric(df[salinity_column], errors="coerce").to_numpy(float)
    if pressure_column and pressure_column in df.columns:
        pres = pd.to_numeric(df[pressure_column], errors="coerce").to_numpy(float)
    else:
        pres = np.zeros_like(ph)

    # par1 = TA (type 1), par2 = pH (type 3) — the same pair your data used
    results = pyco2.sys(
        par1=ta, par1_type=1,
        par2=ph, par2_type=3,
        salinity=sal, temperature=temp, pressure=pres,
        opt_pH_scale=opt_pH_scale,
        opt_k_carbonic=opt_k_carbonic,
        opt_k_bisulfate=opt_k_bisulfate,
        opt_total_borate=opt_total_borate,
    )
    return pd.DataFrame({
        "dic_umol_kg_recomputed": results["dic"],
        "pco2_uatm_recomputed": results["pCO2"],
        "omega_aragonite_recomputed": results["saturation_aragonite"],
        "omega_calcite_recomputed": results["saturation_calcite"],
    })
