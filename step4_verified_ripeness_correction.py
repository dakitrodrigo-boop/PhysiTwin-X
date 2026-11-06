
# ============================================================
# PHYSITWIN-X
# STEP 4: VERIFIED RIPENESS AND FIRMNESS CORRECTIONS
# ============================================================
# Input:
# data/physitwin_x_feature_engineered.csv
#
# Output:
# data/physitwin_x_ripeness_verified.csv
# results/verified_ripeness_correction_log.csv
#
# IMPORTANT:
# Original input is preserved.
# All corrections are based on user-verified observations.
# ============================================================

from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path(
    "data/physitwin_x_feature_engineered.csv"
)

OUTPUT_FILE = Path(
    "data/physitwin_x_ripeness_verified.csv"
)

LOG_FILE = Path(
    "results/verified_ripeness_correction_log.csv"
)

# ------------------------------------------------------------
# 2. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X")
print("STEP 4: VERIFIED DATA CORRECTIONS")
print("=" * 65)

# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print("\nINPUT DATASET")
print("File:", INPUT_FILE)
print("Observations:", original_rows)
print("Mangoes:", df["mango_id"].nunique())

# ------------------------------------------------------------
# 4. REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "peel_color",
    "firmness",
    "ripeness_stage",
    "spoilage_risk"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 5. DUPLICATE CHECK
# ------------------------------------------------------------

if df.duplicated(
    subset=["mango_id", "storage_day"]
).any():
    raise ValueError(
        "Duplicate mango ID and storage day records found."
    )

# ------------------------------------------------------------
# 6. VERIFIED ORIGINAL OBSERVATIONS
# ------------------------------------------------------------
# M10 DAY 13:
# 30 C, 80% RH, 295 g, 2.22% weight loss
# YELLOW ORANGE WITH FORMING FULL BLACK SPOT
# VERY SOFT, OVERRIPE, HIGH RISK
#
# M10 DAY 14:
# 29 C, 90% RH, 290 g, 2.22% weight loss
# BLACK AND BROWN-SPOTTED
# MUSHY, SPOILED, VERY HIGH RISK
#
# M39 DAY 10:
# Ripeness = OVERRIPE
#
# M48 DAY 11:
# Firmness = MUSHY
# Ripeness = SPOILED
# ------------------------------------------------------------

verified_records = [
    {
        "mango_id": "M10",
        "storage_day": 13,
        "ripeness_stage": "OVERRIPE",
        "firmness": "VERY SOFT",
        "temperature_c": 30,
        "relative_humidity_pct": 80,
        "weight_g": 295,
        "weight_loss_pct": 2.22,
        "spoilage_risk": "HIGH RISK"
    },
    {
        "mango_id": "M10",
        "storage_day": 14,
        "ripeness_stage": "SPOILED",
        "firmness": "MUSHY",
        "temperature_c": 29,
        "relative_humidity_pct": 90,
        "weight_g": 290,
        "weight_loss_pct": 2.22,
        "spoilage_risk": "VERY HIGH RISK"
    },
    {
        "mango_id": "M39",
        "storage_day": 10,
        "ripeness_stage": "OVERRIPE",
        "firmness": "VERY SOFT",
        "temperature_c": 29,
        "relative_humidity_pct": 85,
        "weight_g": 255,
        "weight_loss_pct": 2.22,
        "spoilage_risk": "HIGH RISK"
    },
    {
        "mango_id": "M48",
        "storage_day": 11,
        "ripeness_stage": "SPOILED",
        "firmness": "MUSHY",
        "temperature_c": 28,
        "relative_humidity_pct": 86,
        "weight_g": 150,
        "weight_loss_pct": 3.70,
        "spoilage_risk": "VERY HIGH RISK"
    }
]

# ------------------------------------------------------------
# 7. APPLY VERIFIED CORRECTIONS
# ------------------------------------------------------------

correction_log = []

for record in verified_records:

    mango_id = record["mango_id"]
    day = record["storage_day"]

    mask = (
        (df["mango_id"].astype(str) == mango_id)
        &
        (pd.to_numeric(
            df["storage_day"],
            errors="coerce"
        ) == day)
    )

    matching_indices = df.index[mask].tolist()

    if len(matching_indices) != 1:
        raise ValueError(
            f"Expected one record for {mango_id} Day {day}; "
            f"found {len(matching_indices)}."
        )

    index = matching_indices[0]

    previous_values = {}

    # Update only the explicitly verified fields.
    for column in [
        "ripeness_stage",
        "firmness",
        "temperature_c",
        "relative_humidity_pct",
        "weight_g",
        "weight_loss_pct",
        "spoilage_risk"
    ]:

        previous_values[column] = df.at[index, column]

        df.at[index, column] = record[column]

    correction_log.append({
        "mango_id": mango_id,
        "storage_day": day,
        "previous_ripeness_stage":
            previous_values["ripeness_stage"],
        "verified_ripeness_stage":
            record["ripeness_stage"],
        "previous_firmness":
            previous_values["firmness"],
        "verified_firmness":
            record["firmness"],
        "previous_temperature":
            previous_values["temperature_c"],
        "verified_temperature":
            record["temperature_c"],
        "previous_humidity":
            previous_values["relative_humidity_pct"],
        "verified_humidity":
            record["relative_humidity_pct"],
        "previous_weight":
            previous_values["weight_g"],
        "verified_weight":
            record["weight_g"],
        "previous_weight_loss":
            previous_values["weight_loss_pct"],
        "verified_weight_loss":
            record["weight_loss_pct"],
        "previous_spoilage_risk":
            previous_values["spoilage_risk"],
        "verified_spoilage_risk":
            record["spoilage_risk"]
    })

    print(
        f"VERIFIED: {mango_id} Day {day} | "
        f"Ripeness: {record['ripeness_stage']} | "
        f"Firmness: {record['firmness']}"
    )

# ------------------------------------------------------------
# 8. VERIFY ALL CORRECTED VALUES
# ------------------------------------------------------------

for record in verified_records:

    mask = (
        (df["mango_id"].astype(str) == record["mango_id"])
        &
        (df["storage_day"] == record["storage_day"])
    )

    row = df.loc[mask].iloc[0]

    for column in [
        "ripeness_stage",
        "firmness",
        "temperature_c",
        "relative_humidity_pct",
        "weight_g",
        "weight_loss_pct",
        "spoilage_risk"
    ]:

        if row[column] != record[column]:
            raise ValueError(
                f"Validation failed: "
                f"{record['mango_id']} Day "
                f"{record['storage_day']} - {column}"
            )

# ------------------------------------------------------------
# 9. DATASET INTEGRITY VALIDATION
# ------------------------------------------------------------

if len(df) != original_rows:
    raise ValueError(
        "Observation count changed unexpectedly."
    )

if len(df.columns) != original_columns:
    raise ValueError(
        "Column count changed unexpectedly."
    )

if len(df) != 744:
    raise ValueError(
        f"Expected 744 observations, found {len(df)}."
    )

if df["mango_id"].nunique() != 60:
    raise ValueError(
        "Expected 60 unique mangoes."
    )

if df.duplicated(
    subset=["mango_id", "storage_day"]
).any():
    raise ValueError(
        "Duplicate records detected after correction."
    )

# ------------------------------------------------------------
# 10. SAVE VERIFIED DATASET AND CORRECTION LOG
# ------------------------------------------------------------

log_df = pd.DataFrame(correction_log)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

log_df.to_csv(
    LOG_FILE,
    index=False
)

# ------------------------------------------------------------
# 11. FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL VERIFIED CORRECTION REPORT")
print("=" * 65)

print("Total observations:", len(df))
print("Total mangoes:", df["mango_id"].nunique())
print("Total columns:", len(df.columns))
print("Verified records:", len(verified_records))

print("\nVERIFIED OBSERVATIONS")

for record in verified_records:

    print(
        record["mango_id"],
        "| Day:", record["storage_day"],
        "| Temperature:", record["temperature_c"],
        "| RH:", record["relative_humidity_pct"],
        "| Weight:", record["weight_g"],
        "| Weight loss:", record["weight_loss_pct"],
        "| Firmness:", record["firmness"],
        "| Ripeness:", record["ripeness_stage"],
        "| Risk:", record["spoilage_risk"]
    )

print("\nSAVED FILES")
print(OUTPUT_FILE)
print(LOG_FILE)

print("\nOriginal input dataset preserved.")
print("All validation checks passed.")
print("STEP 4 COMPLETED SUCCESSFULLY.")
print("=" * 65)
