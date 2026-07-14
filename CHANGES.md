# ctd_carb_model — three bugs found by running on the real P45_06 output

All three produced plausible-looking output rather than erroring. None would
have surfaced on synthetic data.

Files changed: `ctd_carb_model/data.py`, `predict.py`, `models.py`.
Drop them over your copies.

---

## 1. `predict.py` — only 28 of 983 levels predicted (should be all 983)

**Cause.** The profile-centric table holds the full CTD at every level but the
carbonate columns only at the 28 sampled levels. Predictor resolution was
first-match, and the generic aliases (`salinity`, `depth_m`, `temp_insitu`)
match the **carbonate** columns (28/983) before reaching the CTD-native ones
(`Salinity_PSU`, `Temperature_C`, `Depth_m`, `Oxygen_umol_kg`, all 983/983).
Prediction was therefore restricted to rows that already had samples — the
opposite of a continuous profile.

**Fix.** New `_resolve_on_profile()` picks the alias with the **highest non-null
coverage** rather than first-match. CTD-native spellings added to
`PREDICTOR_ALIASES`. A guard now raises if any predictor still resolves to a
column covering <50% of levels, rather than silently returning a near-empty
field.

## 2. `predict.py` — 955 rows flagged extrapolation, including trained-on casts

**Cause.** The check preferred `cruise_id`, which on the profile table is a
carbonate-side column — populated on 28 rows, NaN on 955. `NaN.isin({...})` is
False, so every unsampled level was marked extrapolation.

**Fix.** Judge extrapolation on the profile's own fully-populated identity
columns, resolved by coverage, preferring **cast** over cruise — a cast with no
bottles of its own is precisely what the model never saw.

**Result:** 983/983 predicted; 209 rows extrapolation on exactly casts 04, 05,
07 — the three with no carbonate bottles, which the merge reconciliation had
independently identified.

## 3. `models.py` — the ANN was untrainable on TA (R² = −1846)

**Cause.** `MLPRegressor` had standardised **inputs** but not the **target**.
With TA ≈ 2300 and `alpha=1.0` L2 regularisation, the net is pulled toward zero
and cannot reach the response magnitude: it predicted **110** for a true mean of
**2305** (bias −2241, i.e. the full size of the response). The MLR-vs-ANN
comparison was therefore meaningless — MLR "won" against a broken opponent,
which defeats the stated purpose of `compare.py`.

**Fix.** `CarbModel` gains `needs_y_scaling`; when set, `fit()` standardises the
target and `predict()` inverts the transform. Enabled on `ANNModel`, off for
`MLRModel` (OLS is scale-invariant and needs no such thing).

**Result:** ANN TA R² −1846 → **0.091**; pH → **0.309**. MLR and ANN are now
essentially tied (pH 0.320 vs 0.309), which is the honest outcome on 38 noisy
samples — the ANN finds no nonlinearity the MLR misses, and `choose_winner()`'s
parsimony rule correctly prefers MLR.

---

## Verified run (P45_06)

```
Model comparison (validation: leave_one_cast_out)

                response model  n     r2    rmse     mae    bias
                ph_total   MLR 38 0.3200  0.0845  0.0654 -0.0016
                ph_total   ANN 38 0.3090  0.0852  0.0659 -0.0010
total_alkalinity_umol_kg   MLR 38 0.1001 49.4876 33.1585  0.5423
total_alkalinity_umol_kg   ANN 38 0.0908 49.7430 33.0367 -0.5074

Winners:
  ph_total: MLR — highest out-of-group R2=0.320
  total_alkalinity_umol_kg: MLR — highest out-of-group R2=0.100
```

13 plots + 4 tables + report + manifest generated cleanly. Example output in
`example_output/`.

---

## Two things to decide (not bugs)

**`data.prepare()` keeps REVIEW rows by default.** `analysis_ready_values =
("PASS", "REVIEW")`. For P45_06, all 16 REVIEW rows are REVIEW *because of*
`replicate_conflict_carried` — so known replicate-conflicted bottles go straight
into training, silently. Consider `("PASS",)` as the modelling default, or run
both and report both. (On P45_06 it changes little: pH R² 0.46 → 0.36 in-sample
on 22 rows — the conflicts are not the whole story.)

**Predictor selection is unstable.** TA's AICc-chosen subset flips from
`salinity` (38 rows) to `temperature + oxygen` (22 rows). With Akaike weights of
0.17–0.24 and 4–7 subsets inside ΔAICc < 2, selection is not identifying real
structure. Report the candidate set.
