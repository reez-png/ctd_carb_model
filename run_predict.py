"""Generate continuous predicted carbonate profiles for P45_06."""
import pandas as pd
from pathlib import Path
from ctd_carb_model.predict import predict_profiles, write_predicted_profiles, summarise

MERGE_OUT = Path(r"C:\Users\OA_2023-03\Projects\merge_tool\ctd_carb_merge\merged_out\data")
OUT_DIR   = Path("outputs/model_products/predicted_profiles")

bottle  = pd.read_csv(MERGE_OUT / "merged_ctd_carbonate.csv", low_memory=False)
profile = pd.read_csv(MERGE_OUT / "merged_ctd_carbonate_full_profile.csv", low_memory=False)

pred, info = predict_profiles(bottle, profile)

out = write_predicted_profiles(pred, OUT_DIR / "P45_06_predicted_carbonate_profiles.csv")
print(summarise(info))
print(f"\nwritten: {out.resolve()}  ({len(pred)} rows)")