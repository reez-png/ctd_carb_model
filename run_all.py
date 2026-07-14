"""
run_all.py — end-to-end modelling run for one cruise.

Reads the ctd_carb_merge master tables, then in order:

  1. prepare   — master table -> model-ready frame (labels measured vs calculated)
  2. compare   — MLR vs ANN through identical leave-one-cast-out folds
  3. report    — plots, tables, markdown report, manifest
  4. predict   — continuous predicted carbonate profiles at every CTD level

Everything lands under OUT_DIR (default outputs/model_products/), which is
gitignored — this script is the versioned artifact, not its output.

Usage:
    python run_all.py
    python run_all.py --merge-dir "C:\\...\\merged_out\\data" --pass-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ctd_carb_model.data import prepare, summarise, DataConfig
from ctd_carb_model.compare import compare_models, format_comparison
from ctd_carb_model.report import generate_all, write_report, write_manifest
from ctd_carb_model.predict import (
    predict_profiles,
    write_predicted_profiles,
    summarise as predict_summarise,
)

# Default location of the merge tool's output. Override with --merge-dir.
DEFAULT_MERGE_DIR = Path(
    r"C:\Users\OA_2023-03\Projects\merge_tool\ctd_carb_merge\merged_out\data"
)
BOTTLE_NAME = "merged_ctd_carbonate.csv"
PROFILE_NAME = "merged_ctd_carbonate_full_profile.csv"

RESPONSES = ["ph_total", "total_alkalinity_umol_kg"]  # measured only — see README


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="End-to-end carbonate modelling run.")
    p.add_argument("--merge-dir", default=str(DEFAULT_MERGE_DIR),
                   help="Directory holding the merge tool's two CSVs.")
    p.add_argument("--out-dir", default="outputs/model_products",
                   help="Where to write plots, tables, report, predictions.")
    p.add_argument("--tag", default="P45_06",
                   help="Prefix for the predicted-profile filename.")
    p.add_argument("--split", default="leave_one_cast_out",
                   choices=["leave_one_cast_out", "leave_one_station_out",
                            "leave_one_cruise_out", "leave_one_season_out"],
                   help="Validation split (grouped, never by row).")
    p.add_argument("--select-by", default="aicc", choices=["aicc", "aic", "bic", "none"],
                   help="Predictor selection criterion; 'none' uses all predictors.")
    p.add_argument("--pass-only", action="store_true",
                   help="Train on PASS rows only, excluding REVIEW (which on "
                        "P45_06 means replicate-conflicted bottles).")
    p.add_argument("--no-predict", action="store_true",
                   help="Skip the continuous-profile step.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    merge_dir = Path(args.merge_dir)
    out_dir = Path(args.out_dir)
    bottle_path = merge_dir / BOTTLE_NAME
    profile_path = merge_dir / PROFILE_NAME

    if not bottle_path.exists():
        print(f"ERROR: bottle table not found: {bottle_path}\n"
              "Run ctd-carb-merge first, or pass --merge-dir.", file=sys.stderr)
        return 1

    # ---- 1. prepare ------------------------------------------------------
    print("=" * 72)
    print("1. PREPARE")
    print("=" * 72)
    bottle = pd.read_csv(bottle_path, low_memory=False)
    config = DataConfig(analysis_ready_values=("PASS",)) if args.pass_only else DataConfig()
    ds = prepare(bottle, config)
    print(summarise(ds))
    if args.pass_only:
        print("\n[--pass-only] REVIEW rows excluded from training.")

    # ---- 2. compare ------------------------------------------------------
    print()
    print("=" * 72)
    print("2. COMPARE  (MLR vs ANN, identical folds)")
    print("=" * 72)
    select_by = None if args.select_by == "none" else args.select_by
    result = compare_models(
        ds.frame, ds.predictors, RESPONSES,
        model_names=["MLR", "ANN"],
        split=args.split,
        select_by=select_by,
    )
    print(format_comparison(result))

    # ---- 3. report -------------------------------------------------------
    print()
    print("=" * 72)
    print("3. REPORT")
    print("=" * 72)
    written = generate_all(result, out_dir, ds.response_status)
    write_report(result, out_dir / "model_report.md", summarise(ds), ds.response_status)
    write_manifest(
        result, out_dir / "manifest.json",
        {"bottle_table": str(bottle_path), "profile_table": str(profile_path),
         "pass_only": args.pass_only},
        summarise(ds),
    )
    print(f"  {len(written['plots'])} plots, {len(written['tables'])} tables")
    print(f"  report:   {(out_dir / 'model_report.md').resolve()}")
    print(f"  manifest: {(out_dir / 'manifest.json').resolve()}")

    # ---- 4. predict continuous profiles ----------------------------------
    if args.no_predict:
        print("\n[--no-predict] skipping continuous profiles.")
        return 0
    if not profile_path.exists():
        print(f"\nWARNING: profile table not found ({profile_path}); "
              "skipping continuous profiles.", file=sys.stderr)
        return 0

    print()
    print("=" * 72)
    print("4. PREDICT  (continuous profiles at every CTD level)")
    print("=" * 72)
    profile = pd.read_csv(profile_path, low_memory=False)
    pred, info = predict_profiles(bottle, profile, select_by=select_by)
    print(predict_summarise(info))
    out = write_predicted_profiles(
        pred, out_dir / "predicted_profiles" / f"{args.tag}_predicted_carbonate_profiles.csv"
    )
    print(f"\n  written: {out.resolve()}  ({len(pred)} rows)")

    # Sanity gate: a continuous profile must cover every CTD level. If this
    # trips, predictors resolved onto sparse carbonate columns (see CHANGES.md).
    n_pred = info.get("n_predicted_rows", 0)
    n_rows = info.get("n_profile_rows", 0)
    if n_rows and n_pred < n_rows:
        print(f"\nWARNING: only {n_pred}/{n_rows} levels predicted — expected all. "
              "Check the profile table's predictor columns.", file=sys.stderr)

    print()
    print("Done. Out-of-group metrics are the honest measure — see README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())