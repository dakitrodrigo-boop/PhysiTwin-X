
from pathlib import Path
import pandas as pd

# ============================================================
# PHYSITWIN-X STEP 9B
# DYNAMIC MANGO STATE CALCULATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = DATA_DIR / "physitwin_x_endpoint_reverified.csv"

OUTPUT_FILE = DATA_DIR / "physitwin_x_dynamic_states.csv"
SUMMARY_FILE = RESULTS_DIR / "step9B_dynamic_state_summary.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X STEP 9B: DYNAMIC STATE CALCULATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "firmness",
    "ripeness_stage",
    "peel_color"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

original_rows = len(df)

# ------------------------------------------------------------
# 2. DEFINE OPERATIONAL ORDINAL STATE SCORES
# These are ordered category codes, not physical measurements.
# ------------------------------------------------------------

firmness_mapping = {
    "VERY FIRM": 0,
    "FIRM": 1,
    "MODERATELY FIRM": 2,
    "MEDIUM FIRM": 3,
    "SLIGHTLY FIRM": 4,
    "SOFT": 5,
    "VERY SOFT": 6,
    "MUSHY": 7
}

ripeness_mapping = {
    "UNRIPE": 0,
    "MATURE GREEN": 1,
    "BREAKER": 2,
    "RIPE": 3,
    "FULLY RIPE": 4,
    "OVERRIPE": 5,
    "SPOILED": 6
}

# Normalize category spacing and capitalization only.
df["firmness"] = (
    df["firmness"].astype(str).str.strip().str.upper()
)

df["ripeness_stage"] = (
    df["ripeness_stage"].astype(str).str.strip().str.upper()
)

# Check for categories not covered by the mappings.
unknown_firmness = sorted(
    set(df["firmness"].dropna()) - set(firmness_mapping)
)

unknown_ripeness = sorted(
    set(df["ripeness_stage"].dropna()) - set(ripeness_mapping)
)

if unknown_firmness or unknown_ripeness:
    raise ValueError(
        "Unmapped categories found.\n"
        f"Firmness: {unknown_firmness}\n"
        f"Ripeness: {unknown_ripeness}"
    )

df["firmness_state_score"] = df["firmness"].map(
    firmness_mapping
)

df["ripeness_state_score"] = df["ripeness_stage"].map(
    ripeness_mapping
)

# ------------------------------------------------------------
# 3. CALCULATE PREVIOUS OBSERVATION VALUES
# ------------------------------------------------------------

grouped = df.groupby("mango_id", sort=False)

previous_columns = [
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "firmness_state_score",
    "ripeness_state_score"
]

for col in previous_columns:
    df[f"previous_{col}"] = grouped[col].shift(1)

# ------------------------------------------------------------
# 4. CALCULATE DYNAMIC STATE CHANGES
# ------------------------------------------------------------

df["elapsed_days"] = (
    df["storage_day"] - df["previous_storage_day"]
)

df["days_since_first_observation"] = (
    df["storage_day"]
    - grouped["storage_day"].transform("min")
)

change_pairs = {
    "temperature_c": "temperature_change_c",
    "relative_humidity_pct": "humidity_change_pct",
    "weight_g": "weight_change_g",
    "weight_loss_pct": "weight_loss_change_pct",
    "firmness_state_score": "firmness_state_change",
    "ripeness_state_score": "ripeness_state_change"
}

for source, destination in change_pairs.items():
    df[destination] = (
        df[source] - df[f"previous_{source}"]
    )

# First observation is the baseline for each mango.
change_columns = list(change_pairs.values())

first_observation = df["previous_storage_day"].isna()

df.loc[first_observation, change_columns] = 0
df.loc[first_observation, "elapsed_days"] = 0

# Calculate changes per elapsed day.
rate_pairs = {
    "weight_change_g": "weight_change_g_per_day",
    "weight_loss_change_pct": "weight_loss_change_pct_per_day",
    "temperature_change_c": "temperature_change_c_per_day",
    "humidity_change_pct": "humidity_change_pct_per_day",
    "firmness_state_change": "firmness_state_change_per_day",
    "ripeness_state_change": "ripeness_state_change_per_day"
}

for change_col, rate_col in rate_pairs.items():
    df[rate_col] = (
        df[change_col] / df["elapsed_days"]
    )

# First observation has no preceding interval.
rate_columns = list(rate_pairs.values())
df.loc[first_observation, rate_columns] = pd.NA

# ------------------------------------------------------------
# 5. VALIDATION
# ------------------------------------------------------------

duplicate_count = df.duplicated(
    subset=["mango_id", "storage_day"]
).sum()

negative_intervals = (
    df["elapsed_days"].dropna() < 0
).sum()

missing_state_scores = df[
    ["firmness_state_score", "ripeness_state_score"]
].isna().sum().sum()

row_count_valid = len(df) == original_rows
mango_count = df["mango_id"].nunique()

validation_passed = (
    row_count_valid
    and duplicate_count == 0
    and negative_intervals == 0
    and missing_state_scores == 0
)

# ------------------------------------------------------------
# 6. SAVE DYNAMIC STATE DATA
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------------------
# 7. CREATE SUMMARY REPORT
# ------------------------------------------------------------

report = [
    "=" * 65,
    "PHYSITWIN-X STEP 9B DYNAMIC STATE SUMMARY",
    "=" * 65,
    f"Total observations: {len(df)}",
    f"Unique mangoes: {mango_count}",
    f"Storage period: Day {df['storage_day'].min()} "
    f"to Day {df['storage_day'].max()}",
    "",
    "DYNAMIC STATE VARIABLES",
    *change_columns,
    *rate_columns,
    "",
    "STATE SCORE DEFINITIONS",
    "Firmness: 0 = VERY FIRM; 7 = MUSHY",
    "Ripeness: 0 = UNRIPE; 6 = SPOILED",
    "These are operational ordinal category codes,",
    "not direct physical measurements.",
    "",
    "VALIDATION",
    f"Original row count preserved: {row_count_valid}",
    f"Duplicate mango-day records: {duplicate_count}",
    f"Negative elapsed intervals: {negative_intervals}",
    f"Missing state scores: {missing_state_scores}",
    f"Validation passed: {validation_passed}",
    "",
    "IMPORTANT NOTES",
    "Each change uses the current and previous observation",
    "for the same mango only.",
    "First observations are treated as individual baselines.",
    "Original peel-color descriptions are preserved.",
    "No future endpoint or spoilage-risk label is used",
    "to calculate the dynamic state changes.",
    "",
    f"Output dataset: {OUTPUT_FILE}",
    "=" * 65
]

SUMMARY_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n".join(report))

if not validation_passed:
    raise ValueError(
        "Step 9B validation failed. Review the summary."
    )

print("\nSTEP 9B COMPLETED SUCCESSFULLY")
