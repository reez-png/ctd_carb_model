# ctd_carb_model — carbonate modelling pipeline

Predicts carbonate chemistry (**pH, TA**) from CTD hydrography, recomputes the
derived variables (**DIC, pCO₂, Ω**) from the predicted pair, and **compares MLR
vs ANN** with leakage-safe validation and full diagnostics.

It reads the **bottle-matched master table** produced by the `ctd_carb_merge`
tool. It does not merge or plot transects — those are separate tools. This one
job: turn matched CTD + carbonate data into honestly-validated models and a
convincing report.

```text
ctd_carb_merge  ──►  master table  ──►  ctd_carb_model  ──►  validated models + comparison + report
```

---

## The five parts (and the notebook for each)

| Part | Module | Notebook | Does |
|---|---|---|---|
| 1 | `data` | `step01_prepare_data` | Load master table; resolve columns; **label measured vs calculated**; drop QC-failed rows |
| 2 | `models` | `step02_fit_and_inspect_models` | MLR + ANN on a **common interface**; PyCO₂SYS recompute for derived vars |
| 2b | `selection` | `step02b_select_predictors_aicc` | **AICc** exhaustive subset ranking; choose predictors per response |
| 3 | `validation` | `step03_validate` | **Leave-one-cast-out** (and cruise/season); honest out-of-group metrics |
| 4 | `compare` | `step04_compare_mlr_ann` | **Fair head-to-head**; one table; reasoned winner |
| 5 | `report` | `step05_diagnostics_and_report` | Predicted-vs-actual, residual, per-fold plots; comparison bars; report + manifest |

---


## Predictor selection (AICc)

By default the pipeline can choose predictors per response by **exhaustive AICc
subset ranking** (`select_by: aicc` in the config, or `compare_models(..., select_by="aicc")`).
AICc is AIC with a small-sample correction — the right criterion at your sample
size. Selection runs **once on the full data** to pick the subset; the chosen
model is then validated **out-of-group** (leave-one-cast-out). Selection answers
"which predictors?"; validation answers "does the chosen model predict?". Both
are reported, and `step02b_select_predictors_aicc` shows the full ranking with
ΔAICc and Akaike weights. If several subsets sit within ΔAICc < 2, the pipeline
says so rather than over-claiming a single winner.

## The scientific principles it enforces

**Measured vs calculated.** pH and TA are measured; DIC, pCO₂, Ω are *calculated*
from the measured pair. So the pipeline only fits models to pH and TA, then
**recomputes** the derived variables from the predicted pH+TA via PyCO₂SYS — the
same way the observed values were made — using the **same carbonate constants**
(read from your data's provenance). This keeps the derived predictions
carbonate-system-consistent and avoids claiming a "DIC model" was validated
against independent DIC.

**Validation, not in-sample fit.** Skill is measured by holding out **whole
casts** (rows from one cast are not independent). A random row split would let a
model "predict" a depth it already saw. Out-of-group R²/RMSE is the only number
that counts; in-sample fit is ignored.

**Fair comparison.** MLR and ANN run through the *same* folds, rows, and metrics,
so the winner reflects the model, not the setup. The winner is declared **with a
reason**, and a simpler model (MLR) is preferred when it's within a small margin
of a complex one (parsimony, and it generalises more safely on small data).

**Honesty guards.** With one cruise, the pipeline refuses leave-one-cruise-out
and says you cannot validate seasonal/temporal transfer. A negative R² is
reported plainly as "worse than predicting the mean." Nothing is dressed up.

> **Expect on a small dataset (tens of samples, one cruise):** MLR will likely
> win and the ANN will struggle — possibly badly (negative R²). That's the
> correct, honest result, not a failure of the pipeline. It's a defensible
> finding: a parsimonious regional MLR predicts shelf pH/TA from CTD; a neural
> network offers no benefit at current sampling. Re-run as data grow.

---

## Install

```powershell
cd ctd_carb_model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"      # pandas, numpy, scikit-learn, PyCO2SYS, matplotlib
python -m pytest -q                    # expect 12 passed
```

Optional extras can be installed alone: `.[ann]` (scikit-learn), `.[recompute]`
(PyCO2SYS), `.[plots]` (matplotlib).

---

## Run it

Set the path to your merge output in `step01`, then run the notebooks in order
(`step01` → `step05`). Or from Python:

```python
from ctd_carb_model import data, compare, report
import pandas as pd

ds = data.prepare(pd.read_csv("merged_ctd_carbonate.csv"))
measured = [r for r in ds.responses if ds.response_status[r] == "measured"]
result = compare.compare_models(ds.frame, ds.predictors, measured, ["MLR", "ANN"])
print(compare.format_comparison(result))
report.generate_all(result, "outputs/model_products", ds.response_status)
```

See `COLUMN_NAMES.md` for the exact input columns and how they're resolved.

---

## Outputs

```text
outputs/model_products/
  model_comparison_table.csv          MLR vs ANN, every response, honest metrics
  winners.csv                         per-response winner + reason
  comparison_bars.png                 R² / RMSE side by side
  <response>_<model>_pred_vs_actual.png   the headline plots
  <response>_<model>_residuals.png
  <response>_<model>_per_fold.png
  <response>_<model>_out_of_group_predictions.csv
  report.md                           written summary
  manifest.json                       reproducibility record
```

---

## Scope

Modelling only. Merging is `ctd_carb_merge`; observed profile figures and 2D
transect sections are the plotting tool. Keeping them separate (at the
supervisor's direction) means each is simple, tested, and publishable on its own.
