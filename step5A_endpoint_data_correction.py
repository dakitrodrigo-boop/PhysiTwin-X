
import pandas as pd
from pathlib import Path

# ==================================================
# PHYSITWIN-X STEP 5A
# VERIFIED ENDPOINT DATA CORRECTION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = DATA_DIR / "physitwin_x_ripeness_verified.csv"
OUTPUT_FILE = DATA_DIR / "physitwin_x_endpoint_verified.csv"
LOG_FILE = RESULTS_DIR / "endpoint_correction_log.csv"

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = [
    "mango_id", "storage_day", "temperature_c",
    "relative_humidity_pct", "weight_g",
    "weight_loss_pct", "peel_color",
    "firmness", "ripeness_stage", "spoilage_risk"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["storage_day"] = pd.to_numeric(
    df["storage_day"], errors="raise"
)

df["mango_id"] = df["mango_id"].astype(str).str.strip()

if df.duplicated(["mango_id", "storage_day"]).any():
    raise ValueError("Duplicate mango/day records detected.")

log = []

def get_row(mango, day):
    match = df[
        (df["mango_id"] == mango)
        & (df["storage_day"] == day)
    ]

    if len(match) != 1:
        raise ValueError(
            f"Expected exactly one record for {mango} Day {day}; "
            f"found {len(match)}."
        )

    return match.index[0]


def correct_value(mango, day, column, expected, corrected):
    idx = get_row(mango, day)
    old_value = str(df.at[idx, column]).strip()

    if old_value != expected:
        raise ValueError(
            f"Unexpected value for {mango} Day {day}, {column}: "
            f"{old_value!r}. Expected {expected!r}. "
            "No output has been saved."
        )

    df.at[idx, column] = corrected

    log.append({
        "mango_id": mango,
        "storage_day": day,
        "column": column,
        "previous_value": old_value,
        "verified_value": corrected,
        "action": "CORRECTED"
    })


# --------------------------------------------------
# 1. M01 DAY 10
# --------------------------------------------------

correct_value("M01", 10, "firmness", "SPOILED", "MUSHY")
correct_value("M01", 10, "ripeness_stage", "MUSHY", "SPOILED")


# --------------------------------------------------
# 2. M21 DAY 12
# --------------------------------------------------

correct_value("M21", 12, "ripeness_stage", "OVERRIPE", "SPOILED")


# --------------------------------------------------
# 3. M25 DAY 13
# --------------------------------------------------

correct_value("M25", 13, "ripeness_stage", "OVERRIPE", "SPOILED")


# --------------------------------------------------
# 4. M40 DAY 14
# ADD VERIFIED ORIGINAL MEASUREMENT
# --------------------------------------------------

if not (
    (df["mango_id"] == "M40")
    & (df["storage_day"] == 14)
).any():

    previous = df[
        (df["mango_id"] == "M40")
        & (df["storage_day"] == 13)
    ]

    if len(previous) != 1:
        raise ValueError("Cannot locate M40 Day 13 base record.")

    new_row = previous.iloc[0].copy()

    verified_values = {
        "mango_id": "M40",
        "storage_day": 14,
        "temperature_c": 29.0,
        "relative_humidity_pct": 90.0,
        "weight_g": 305.0,
        "weight_loss_pct": 2.22,
        "peel_color": "BLACK SPOTTED WITH BROWN SPOT",
        "firmness": "MUSHY",
        "ripeness_stage": "SPOILED",
        "spoilage_risk": "VERY HIGH RISK"
    }

    for column, value in verified_values.items():
        new_row[column] = value

    # Preserve original-value columns when present.
    if "peel_color_original" in df.columns:
        new_row["peel_color_original"] = verified_values["peel_color"]

    if "firmness_original" in df.columns:
        new_row["firmness_original"] = verified_values["firmness"]

    # Do not inherit a review flag from Day 13.
    if "data_review_required" in df.columns:
        new_row["data_review_required"] = False

    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True
    )

    log.append({
        "mango_id": "M40",
        "storage_day": 14,
        "column": "Complete observation",
        "previous_value": "Record absent",
        "verified_value": "Original Day 14 measurement",
        "action": "ADDED VERIFIED OBSERVATION"
    })

else:
    raise ValueError(
        "M40 Day 14 already exists. Check it manually "
        "before adding another record."
    )


# --------------------------------------------------
# 5. Recalculate sequential features
# --------------------------------------------------

df = df.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

grouped = df.groupby("mango_id", sort=False)

if "days_since_first_observation" in df.columns:
    df["days_since_first_observation"] = (
        df["storage_day"]
        - grouped["storage_day"].transform("min")
    )

previous_day = grouped["storage_day"].shift(1)
day_gap = df["storage_day"] - previous_day

if "observation_gap_days" in df.columns:
    df["observation_gap_days"] = day_gap

if "weight_change_g" in df.columns:
    df["weight_change_g"] = grouped["weight_g"].diff()

if "weight_loss_change_pct" in df.columns:
    df["weight_loss_change_pct"] = (
        grouped["weight_loss_pct"].diff()
    )

if "weight_loss_rate_pct_per_day" in df.columns:
    df["weight_loss_rate_pct_per_day"] = (
        grouped["weight_loss_pct"].diff() / day_gap
    )

if "temperature_change_c" in df.columns:
    df["temperature_change_c"] = (
        grouped["temperature_c"].diff()
    )

if "humidity_change_pct" in df.columns:
    df["humidity_change_pct"] = (
        grouped["relative_humidity_pct"].diff()
    )


# --------------------------------------------------
# 6. Final validation
# --------------------------------------------------

if df.duplicated(["mango_id", "storage_day"]).any():
    raise ValueError("Duplicate records found after correction.")

if len(df) != 745:
    raise ValueError(
        f"Expected 745 observations after adding M40 Day 14; "
        f"found {len(df)}. Check the dataset."
    )

# --------------------------------------------------
# 7. Save corrected dataset and audit log
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)
pd.DataFrame(log).to_csv(LOG_FILE, index=False)

print("\n" + "=" * 55)
print("PHYSITWIN-X STEP 5A CORRECTION REPORT")
print("=" * 55)

print("Total observations:", len(df))
print("Total mangoes:", df["mango_id"].nunique())
print("Corrections and additions:", len(log))

print("\nVERIFIED ENDPOINT RECORDS")

for mango, day in [
    ("M01", 10),
    ("M21", 12),
    ("M25", 13),
    ("M40", 14)
]:
    row = df[
        (df["mango_id"] == mango)
        & (df["storage_day"] == day)
    ]

    print(
        mango,
        "Day", day,
        "| Firmness:", row.iloc[0]["firmness"],
        "| Ripeness:", row.iloc[0]["ripeness_stage"]
    )

print("\nCorrected dataset:", OUTPUT_FILE)
print("Correction log:", LOG_FILE)

print("\nOriginal input file preserved.")
print("Step 5A completed.")
