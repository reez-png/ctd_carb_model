# COLUMN_NAMES.md — modelling pipeline input reference

The modelling pipeline reads the **bottle-matched master table** from
`ctd_carb_merge`. It resolves columns through alias maps, so you do **not** need
to rename your merge output — the real oa_pipeline / CTD spellings are
recognised. This file lists what it looks for and what it does with each.

---

## Predictors (the CTD physical variables it regresses on)

| Canonical name (used internally) | Accepted spellings in your data |
|---|---|
| `ctd_temperature_c` | `ctd_temperature_c`, `temperature_degC`, `temp_insitu`, `temperature_insitu_c`, `calc_temperature_c` |
| `ctd_salinity` | `ctd_salinity`, `salinity_psu`, `salinity`, `sal`, `calc_salinity` |
| `ctd_oxygen` | `ctd_oxygen`, `oxygen_umol_kg`, `oxygen_umol_l`, `dissolved_oxygen_umol_L` |
| `ctd_depth_m` | `ctd_depth_m`, `depth_m`, `depth_round_m`, `depth_teos10_m` |
| `ctd_sigma_theta` (optional) | `sigma_theta_kg_m3`, `sigma0_kg_m3` |
| `ctd_pressure_dbar` (for recompute) | `pressure_dbar`, `pressure_insitu_dbar` |

These come from the merge tool, which writes the CTD values with a `ctd_` prefix.
In your real data, oxygen arrives as `oxygen_umol_l` — the alias map finds it and
maps it to `ctd_oxygen` automatically.

You set which predictors to use in `config/model_config.yaml` (`predictors:`).
Default: temperature, salinity, oxygen, depth.

---

## Responses (what the models predict) — and their status

| Canonical | Status | Accepted spellings | How it's predicted |
|---|---|---|---|
| `ph_total` | **measured** | `ph_total`, `ph_best`, `ph_observed`, `ph_co2sys` | modelled directly (MLR/ANN) |
| `total_alkalinity_umol_kg` | **measured** | `total_alkalinity_umol_kg`, `ta_best_umolkg`, `ta_umol_kg`, `ta` | modelled directly (MLR/ANN) |
| `dic_umol_kg` | calculated | `dic_best_umol_kg`, `dic_calculated_umol_kg`, `dic` | **recomputed** from predicted pH+TA |
| `pco2_uatm` | calculated | `pco2_best_uatm`, `pco2_calc_uatm` | **recomputed** from predicted pH+TA |
| `omega_aragonite` | calculated | `omega_aragonite_calc`, `omega_ar` | **recomputed** from predicted pH+TA |
| `omega_calcite` | calculated | `omega_calcite_calc`, `omega_ca` | **recomputed** from predicted pH+TA |

**The key rule:** only the **measured** variables (pH, TA) are fit by a model.
The **calculated** ones are obtained by recomputing from the predicted pH+TA via
PyCO₂SYS, the same way your observed values were derived. The pipeline labels
each response so this never gets blurred in a report.

---

## Grouping columns (for leakage-safe validation)

| Logical role | Accepted spellings | Used for |
|---|---|---|
| `cast_id` | `cast_id`, `cast`, `castno` | leave-one-cast-out (always available) |
| `cruise_id` | `cruise_id`, `cruise` | leave-one-cruise-out (needs ≥2 cruises) |
| `station` | `station`, `station_id` | leave-one-station-out |
| `season` | `season`, `upwelling_phase` | leave-one-season-out (needs ≥2 seasons) |

With a single cruise, only leave-one-cast-out runs; the pipeline refuses the
others and tells you so, rather than fabricating a score.

---

## QC filtering

If the master table carries the carbonate pipeline's audit column
(`analysis_audit_status`), the pipeline keeps only analysis-ready rows
(`PASS`, `REVIEW`) and drops `FAIL`. It also drops rows with `flag_no_ctd_match`
(no predictors available). Both are configurable in `model_config.yaml`.

---

## Carbonate constants for the recompute

The recompute step must use the **same constants** your observed DIC/pCO₂/Ω were
computed with, or predicted-vs-observed comparisons on those variables aren't
like-for-like. Your merged data carries the provenance to set these:

| Provenance column in your data | PyCO₂SYS option |
|---|---|
| `carbonate_ph_scale` | `opt_pH_scale` (1 = total) |
| `carbonate_constants` | `opt_k_carbonic` (10 = Lueker 2000) |
| `carbon_input_pair_used` | confirms TA+pH input pair |

Set them in `model_config.yaml` under `recompute:`. Defaults: total scale,
Lueker et al. 2000.

---

## Minimal example of the expected input

A master table with at least these columns will run end-to-end:

```text
cast_id, cruise_id,
ctd_temperature_c, ctd_salinity, ctd_oxygen (or oxygen_umol_l), ctd_depth_m,
ph_best, ta_best_umolkg,
analysis_audit_status   (optional, for QC filtering)
```

Anything extra (the dozens of provenance/flag columns your Stage-4 file carries)
is passed through untouched and ignored unless needed.
