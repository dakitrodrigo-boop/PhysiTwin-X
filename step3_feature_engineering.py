
# ============================================================
# PHYSITWIN-X
# STEP 3: FEATURE ENGINEERING
# ============================================================
# Purpose:
# Generate additional time-dependent features from verified
# mango observations without removing or changing original rows.
#
# Input:
# data/physitwin_x_verified_cleaned_v2.csv
#
# Output:
# data/physitwin_x_feature_engineered.csv
# results/step3_feature_engineering_summary.txt
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path("data/physitwin_x_verified_cleaned_v2.csv")
OUTPUT_FILE = Path("data/physitwin_x_feature_engineered.csv")
SUMMARY_FILE = Path("results/step3_feature_engineering_summary.txt")

# ------------------------------------------------------------
# 2. CREATE OUTPUT FOLDER
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X")
print("STEP 3: FEATURE ENGINEERING")
print("=" * 65)

# ------------------------------------------------------------
# 3. LOAD VERIFIED DATASET
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

initial_rows = len(df)

print("\nINPUT DATASET")
print("File:", INPUT_FILE)
print("Observations:", initial_rows)

required_columns = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 4. CHECK ORIGINAL DATA
# ------------------------------------------------------------

if df["mango_id"].isna().any():
    raise ValueError("Missing mango_id values detected.")

if df["storage_day"].isna().any():
    raise ValueError("Missing storage_day values detected.")

if df.duplicated(
    subset=["mango_id", "storage_day"]
).any():
    raise ValueError(
        "Duplicate mango_id and storage_day records detected."
    )

df["storage_day"] = pd.to_numeric(
    df["storage_day"], errors="raise"
)

for col in [
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 5. SORT OBSERVATIONS
# ------------------------------------------------------------

df = df.sort_values(
    by=["mango_id", "storage_day"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ------------------------------------------------------------

grouped = df.groupby("mango_id", sort=False)

# A. Days since first recorded observation
df["days_since_first_observation"] = (
    df["storage_day"]
    - grouped["storage_day"].transform("min")
)

# B. Observation gap between consecutive records
df["observation_gap_days"] = grouped[
    "storage_day"
].diff()

# C. Change in weight from previous observation
df["weight_change_g"] = grouped[
    "weight_g"
].diff()

# D. Change in weight-loss percentage
df["weight_loss_change_pct"] = grouped[
    "weight_loss_pct"
].diff()

# E. Weight-loss rate per day
valid_gap = df["observation_gap_days"].replace(0, np.nan)

df["weight_loss_rate_pct_per_day"] = (
    df["weight_loss_change_pct"] / valid_gap
)

# F. Temperature change from previous observation
df["temperature_change_c"] = grouped[
    "temperature_c"
].diff()

# G. Relative humidity change from previous observation
df["humidity_change_pct"] = grouped[
    "relative_humidity_pct"
].diff()

# ------------------------------------------------------------
# 7. CHECK ENGINEERED FEATURES
# ------------------------------------------------------------

engineered_features = [
    "days_since_first_observation",
    "observation_gap_days",
    "weight_change_g",
    "weight_loss_change_pct",
    "weight_loss_rate_pct_per_day",
    "temperature_change_c",
    "humidity_change_pct"
]

# First observations naturally have no previous-record changes.
first_observations = grouped.cumcount() == 0

for feature in [
    "observation_gap_days",
    "weight_change_g",
    "weight_loss_change_pct",
    "weight_loss_rate_pct_per_day",
    "temperature_change_c",
    "humidity_change_pct"
]:
    unexpected_first_values = df.loc[
        first_observations, feature
    ].notna().sum()

    if unexpected_first_values:
        raise ValueError(
            f"Unexpected values in first-observation feature: {feature}"
        )

# ------------------------------------------------------------
# 8. DATASET VALIDATION
# ------------------------------------------------------------

final_rows = len(df)
total_mangoes = df["mango_id"].nunique()

if final_rows != initial_rows:
    raise ValueError(
        "Row count changed during feature engineering!"
    )

if total_mangoes != 60:
    print(
        f"WARNING: Expected 60 mangoes, found {total_mangoes}."
    )

if final_rows != 744:
    print(
        f"WARNING: Expected 744 observations, found {final_rows}."
    )

# Count missing values in engineered features.
missing_feature_counts = df[
    engineered_features
].isna().sum()

# ------------------------------------------------------------
# 9. SAVE ENGINEERED DATASET
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------------------
# 10. GENERATE SUMMARY REPORT
# ------------------------------------------------------------

summary = []

summary.append("=" * 65)
summary.append("PHYSITWIN-X STEP 3 SUMMARY")
summary.append("=" * 65)

summary.append(f"Input file: {INPUT_FILE}")
summary.append(f"Output file: {OUTPUT_FILE}")
summary.append(f"Total observations: {final_rows}")
summary.append(f"Total mangoes: {total_mangoes}")
summary.append(f"Total columns: {len(df.columns)}")

summary.append("\nENGINEERED FEATURES")

for feature in engineered_features:
    summary.append(
        f"{feature}: {missing_feature_counts[feature]} missing values"
    )

summary.append("\nFEATURE DEFINITIONS")
summary.append(
    "1. Days since first observation: elapsed storage days "
    "from each mango's first recorded observation."
)
summary.append(
    "2. Observation gap: days between consecutive observations."
)
summary.append(
    "3. Weight change: current weight minus previous recorded weight."
)
summary.append(
    "4. Weight-loss change: current percentage minus previous percentage."
)
summary.append(
    "5. Weight-loss rate: weight-loss percentage change divided "
    "by observation gap."
)
summary.append(
    "6. Temperature change: current temperature minus previous temperature."
)
summary.append(
    "7. Humidity change: current relative humidity minus previous humidity."
)

summary.append("\nVALIDATION")
summary.append(f"Original row count: {initial_rows}")
summary.append(f"Final row count: {final_rows}")
summary.append(
    "Row count preserved: " + str(initial_rows == final_rows)
)
summary.append(
    "Duplicate mango/day records: "
    + str(df.duplicated(["mango_id", "storage_day"]).sum())
)

summary.append("\nSTEP 3 COMPLETED.")

SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 11. DISPLAY RESULTS
# ------------------------------------------------------------

print("\nENGINEERED FEATURES")

for feature in engineered_features:
    print(
        f"{feature}: "
        f"{missing_feature_counts[feature]} missing values"
    )

print("\nFINAL DATASET INFORMATION")
print("Total observations:", final_rows)
print("Total mangoes:", total_mangoes)
print("Total columns:", len(df.columns))

print("\nSAVED FILES")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\n" + "=" * 65)
print("STEP 3 COMPLETED SUCCESSFULLY.")
print("=" * 65)
