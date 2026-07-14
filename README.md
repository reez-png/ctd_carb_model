# ctd_carb_model — predict carbonate chemistry from CTD hydrography

Reads the bottle-matched master table from **`ctd_carb_merge`** and produces
carbonate predictions with leakage-safe validation, an MLR-vs-ANN comparison,
and continuous predicted profiles.

```text
ctd_carb_merge ──► merged_ctd_carbonate.csv          ──► train (Parts 1-5)
               └─► merged_ctd_carbonate_full_profile.csv ──► predict (Part 6)
```

For input column names see **`MODEL_COLUMN_NAMES.md`**.

---

## The design rule this project runs on

Only the **measured** variables (pH, TA) are fitted by a model. The
**calculated** ones (DIC, pCO₂, Ω) are obtained by recomputing from the
predicted pH+TA via PyCO₂SYS — the same way the observed values were derived —
so derived predictions stay carbonate-system-consistent and are never passed off
as independent measurements.

Validation splits by **group, never by row**. Rows from one cast are not
independent (a profile is smooth), so a random row split lets a model "predict"
water it already saw. Holding out whole casts/cruises/seasons is the real test.

---

## The pipeline

| Part | Module | Does |
|---|---|---|
| 1 | `data.py` | master table → model-ready frame; labels measured vs calculated; QC filter |
| 2 | `models.py` | `MLRModel`, `ANNModel` behind one interface; PyCO₂SYS recompute |
| 3 | `validation.py` | leave-one-cast/station/cruise/season-out; honest out-of-group metrics |
| 4 | `compare.py` | MLR vs ANN through identical folds; explicit winner rule |
| 5 | `report.py` | predicted-vs-actual, residual, per-fold plots; markdown report + manifest |
| 6 | `predict.py` | train on bottles → predict every CTD level → recompute derived → flag extrapolation |
| — | `plotting/` | observed profiles, transect sections, relationships, labels |

## Install

```powershell
cd ctd_carb_model
python -m pip install -e ".[all]"
```

## Run it — the whole pipeline

`run_all.py` does prepare → compare → report → predict in one go:

```powershell
python run_all.py
```

Point it at your merge output and name the cruise:

```powershell
python run_all.py `
  --merge-dir "C:\...\merge_tool\ctd_carb_merge\merged_out\data" `
  --tag P45_06
```

| flag | default | does |
|---|---|---|
| `--merge-dir` | the P45_06 merged_out path | where the merge tool's two CSVs live |
| `--out-dir` | `outputs/model_products` | where plots, tables, report, predictions land |
| `--tag` | `P45_06` | prefix for the predicted-profile filename |
| `--split` | `leave_one_cast_out` | validation split (grouped, never by row) |
| `--select-by` | `aicc` | predictor selection criterion; `none` uses all |
| `--pass-only` | off | train on PASS rows only, excluding REVIEW |
| `--no-predict` | off | skip the continuous-profile step |

**What it writes** (all under `--out-dir`, which is gitignored — the script is
the versioned artifact, not its output):

```text
outputs/model_products/
├── model_report.md                     ← read this first
├── manifest.json                       ← inputs, params, results, for reproducibility
├── model_comparison_table.csv          ← MLR vs ANN, out-of-group metrics
├── winners.csv                         ← per-response winner + stated reason
├── <response>_predictor_selection.csv  ← full AICc ranking, all subsets
├── *_pred_vs_actual.png, *_residuals.png, *_per_fold.png, comparison_bars.png
├── *_out_of_group_predictions.csv      ← every held-out prediction
└── predicted_profiles/
    └── <tag>_predicted_carbonate_profiles.csv   ← the continuous field
```

**The check that matters:** the predict stage must report `rows predicted: N of
N` — every CTD level. If it predicts only the sampled levels, predictors have
resolved onto sparse carbonate columns and the run warns you (see `CHANGES.md`).

### The `--pass-only` question

`data.prepare()` keeps `PASS` **and** `REVIEW` rows by default. On P45_06 every
REVIEW row is REVIEW *because of* `replicate_conflict_carried` — so conflicted
bottles train silently. Run both and report both:

```powershell
python run_all.py --out-dir outputs/model_products            # all 38
python run_all.py --out-dir outputs/model_products_pass_only --pass-only  # 22
```

On P45_06 excluding them does not help — pH out-of-group R² falls from 0.32 to
0.06 — because the cost in sample size exceeds the gain in precision. That is
itself worth reporting.

## Use it from Python

```python
import pandas as pd
from ctd_carb_model.data import prepare, summarise
from ctd_carb_model.compare import compare_models, format_comparison

bottle = pd.read_csv("merged_out/data/merged_ctd_carbonate.csv")
ds = prepare(bottle)
print(summarise(ds))

result = compare_models(
    ds.frame, ds.predictors,
    ["ph_total", "total_alkalinity_umol_kg"],
    model_names=["MLR", "ANN"],
    split="leave_one_cast_out",
    select_by="aicc",
)
print(format_comparison(result))
```

### Continuous predicted profiles

```python
from ctd_carb_model.predict import predict_profiles, write_predicted_profiles, summarise as psum

profile = pd.read_csv("merged_out/data/merged_ctd_carbonate_full_profile.csv")
pred, info = predict_profiles(bottle, profile)
print(psum(info))
write_predicted_profiles(pred, "outputs/model_products/predicted_profiles/P45_06.csv")
```

Trains on the bottle table, predicts pH/TA at **every** CTD level, recomputes
DIC/pCO₂/Ω from the predicted pair, and sets `is_extrapolation` on casts that
had no carbonate bottles of their own.

---

## Reading the output honestly

- **Out-of-group R² is the only number that counts.** In-sample fit is not skill.
- **A negative R² means the model is worse than predicting the mean.** That is a
  result, not a bug — it says the data cannot support a model yet.
- **Read the per-fold table, not just the pooled metric.** A pooled R² can hide
  folds ranging from good to catastrophic.
- **ΔAICc < 2 means "not decisively best".** If several predictor subsets sit
  inside that window, report the candidate set rather than the winner.
  `selection.interpret()` says so in words.
- **`is_extrapolation` rows rest on no local samples.** Keep them out of any
  figure that reads as observation.

---

## Verified on real data (P45_06, July 2026)

Run end-to-end against the ctd_carb_merge v0.3.0 output: 38 bottles, 983 CTD
levels, 8 casts, 1 cruise.

| | result |
|---|---|
| model-ready rows | 38 (4 predictors: temperature, salinity, oxygen, depth) |
| predicted profile levels | 983 / 983 |
| extrapolation | 209 rows on casts 04, 05, 07 (no bottles of their own) |
| pH out-of-group R² | MLR 0.320 · ANN 0.309 |
| TA out-of-group R² | MLR 0.100 · ANN 0.091 |
| winner | MLR for both (parsimony; ANN finds no nonlinearity) |

**These models have little demonstrated predictive skill, and the pipeline says
so.** The limiting factor is measurement precision, not model form: CRM
reproducibility for TA batch 213 was ±35 µmol/kg against an expected ±3, traced
to a documented titration protocol deviation. Predicted profiles from this
cruise are a method demonstration, not a validated carbonate field.

See `CHANGES.md` for the three bugs this real-data run exposed.
