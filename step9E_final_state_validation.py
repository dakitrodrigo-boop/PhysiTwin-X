
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PHYSITWIN-X STEP 9E
# FINAL DYNAMIC STATE VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SOURCE_FILE = DATA_DIR / "physitwin_x_endpoint_reverified.csv"
DYNAMIC_FILE = DATA_DIR / "physitwin_x_dynamic_states.csv"
DAILY_FILE = DATA_DIR / "physitwin_x_daily_state_summary.csv"

REPORT_FILE = RESULTS_DIR / "step9E_final_validation.txt"
GAPS_FILE = RESULTS_DIR / "step9E_observation_gaps.csv"

print("=" * 65)
print("PHYSITWIN-X STEP 9E: FINAL STATE VALIDATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------

source = pd.read_csv(SOURCE_FILE)
dynamic = pd.read_csv(DYNAMIC_FILE)
daily = pd.read_csv(DAILY_FILE)

source = source.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

dynamic = dynamic.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 2. CHECK DATASET INTEGRITY
# ------------------------------------------------------------

checks = {}

checks["Source contains 745 observations"] = len(source) == 745
checks["Dynamic dataset contains 745 observations"] = len(dynamic) == 745

checks["Both datasets contain 60 mangoes"] = (
    source["mango_id"].nunique() == 60
    and dynamic["mango_id"].nunique() == 60
)

checks["No duplicate source mango-day records"] = (
    source.duplicated(["mango_id", "storage_day"]).sum() == 0
)

checks["No duplicate dynamic mango-day records"] = (
    dynamic.duplicated(["mango_id", "storage_day"]).sum() == 0
)

# ------------------------------------------------------------
# 3. VERIFY ORIGINAL OBSERVATIONS WERE PRESERVED
# ------------------------------------------------------------

source_columns = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "peel_color",
    "firmness",
    "ripeness_stage"
]

merged = source[source_columns].merge(
    dynamic[source_columns],
    on=["mango_id", "storage_day"],
    how="outer",
    suffixes=("_source", "_dynamic"),
    indicator=True
)

checks["All source records exist in dynamic dataset"] = (
    (merged["_merge"] == "both").all()
)

for col in source_columns[2:]:
    left = merged[f"{col}_source"]
    right = merged[f"{col}_dynamic"]

    if pd.api.types.is_numeric_dtype(left):
        equal = np.isclose(
            left.astype(float),
            right.astype(float),
            equal_nan=True
        ).all()
    else:
        equal = (
            left.fillna("<NA>") == right.fillna("<NA>")
        ).all()

    checks[f"Original {col} preserved"] = bool(equal)

# ------------------------------------------------------------
# 4. RECOMPUTE STATE CHANGES INDEPENDENTLY
# ------------------------------------------------------------

test = dynamic.copy()

grouped = test.groupby("mango_id", sort=False)

test["check_previous_day"] = grouped["storage_day"].shift(1)

change_pairs = {
    "temperature_c": "temperature_change_c",
    "relative_humidity_pct": "humidity_change_pct",
    "weight_g": "weight_change_g",
    "weight_loss_pct": "weight_loss_change_pct",
    "firmness_state_score": "firmness_state_change",
    "ripeness_state_score": "ripeness_state_change"
}

interval = (
    test["storage_day"] - test["check_previous_day"]
)

first = test["check_previous_day"].isna()

checks["First observations have zero elapsed interval"] = (
    (test.loc[first, "elapsed_days"] == 0).all()
)

checks["Elapsed intervals match observed day differences"] = (
    np.isclose(
        test.loc[~first, "elapsed_days"],
        interval.loc[~first]
    ).all()
)

for source_col, change_col in change_pairs.items():

    previous = test.groupby("mango_id", sort=False)[
        source_col
    ].shift(1)

    expected = test[source_col] - previous
    expected.loc[first] = 0

    actual = test[change_col]

    checks[f"{change_col} calculation is correct"] = (
        np.isclose(
            actual.astype(float),
            expected.astype(float),
            equal_nan=True
        ).all()
    )

# ------------------------------------------------------------
# 5. VERIFY DAILY RATES
# ------------------------------------------------------------

rate_pairs = {
    "weight_change_g": "weight_change_g_per_day",
    "weight_loss_change_pct": "weight_loss_change_pct_per_day",
    "temperature_change_c": "temperature_change_c_per_day",
    "humidity_change_pct": "humidity_change_pct_per_day",
    "firmness_state_change": "firmness_state_change_per_day",
    "ripeness_state_change": "ripeness_state_change_per_day"
}

for change_col, rate_col in rate_pairs.items():

    expected_rate = (
        test[change_col] / test["elapsed_days"]
    )

    expected_rate.loc[first] = np.nan

    checks[f"{rate_col} calculation is correct"] = (
        np.isclose(
            test[rate_col].astype(float),
            expected_rate.astype(float),
            equal_nan=True
        ).all()
    )

# ------------------------------------------------------------
# 6. CHECK ORDINAL STATE CODES
# ------------------------------------------------------------

checks["Firmness scores stay between 0 and 7"] = (
    dynamic["firmness_state_score"].between(0, 7).all()
)

checks["Ripeness scores stay between 0 and 6"] = (
    dynamic["ripeness_state_score"].between(0, 6).all()
)

# ------------------------------------------------------------
# 7. RECOMPUTE DAILY OBSERVATION COUNTS AND MEANS
# ------------------------------------------------------------

recomputed = (
    dynamic.groupby("storage_day")
    .agg(
        mangoes_observed=("mango_id", "nunique"),
        observations=("mango_id", "size"),
        mean_firmness_state=("firmness_state_score", "mean"),
        mean_ripeness_state=("ripeness_state_score", "mean"),
        mean_weight_loss_pct=("weight_loss_pct", "mean"),
        mean_weight_g=("weight_g", "mean"),
        mean_temperature_c=("temperature_c", "mean"),
        mean_humidity_pct=("relative_humidity_pct", "mean")
    )
    .reset_index()
)

daily_comparison = recomputed.merge(
    daily,
    on="storage_day",
    suffixes=("_recomputed", "_saved")
)

daily_checks = [
    "mangoes_observed",
    "observations",
    "mean_firmness_state",
    "mean_ripeness_state",
    "mean_weight_loss_pct",
    "mean_weight_g",
    "mean_temperature_c",
    "mean_humidity_pct"
]

for col in daily_checks:

    checks[f"Daily {col} matches recomputation"] = (
        np.isclose(
            daily_comparison[f"{col}_recomputed"],
            daily_comparison[f"{col}_saved"],
            equal_nan=True
        ).all()
    )

# ------------------------------------------------------------
# 8. IDENTIFY WITHIN-MANGO OBSERVATION GAPS
# ------------------------------------------------------------

gap_data = dynamic[
    ["mango_id", "storage_day"]
].copy()

gap_data["previous_day"] = (
    gap_data.groupby("mango_id")["storage_day"].shift(1)
)

gap_data["gap_days"] = (
    gap_data["storage_day"] - gap_data["previous_day"]
)

gaps = gap_data[
    gap_data["gap_days"] > 1
].copy()

gaps.to_csv(GAPS_FILE, index=False)

# ------------------------------------------------------------
# 9. FINAL VALIDATION REPORT
# ------------------------------------------------------------

passed_count = sum(checks.values())
failed_count = len(checks) - passed_count

report = [
    "=" * 65,
    "PHYSITWIN-X STEP 9E FINAL VALIDATION REPORT",
    "=" * 65,
    f"Total validation checks: {len(checks)}",
    f"Passed: {passed_count}",
    f"Failed: {failed_count}",
    "",
    "VALIDATION DETAILS"
]

for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    report.append(f"{status}: {name}")

report.extend([
    "",
    "OBSERVATION GAP ANALYSIS",
    f"Within-mango gaps greater than one day: {len(gaps)}",
    f"Maximum observed interval: {dynamic['elapsed_days'].max()} days",
    "",
    "IMPORTANT INTERPRETATION",
    "Observation gaps are reported for transparency.",
    "They are not automatically treated as errors.",
    "State changes are calculated between consecutive",
    "available observations for each individual mango.",
    "",
    "OUTPUT FILE",
    str(GAPS_FILE),
    "",
    f"FINAL VALIDATION: {'PASSED' if failed_count == 0 else 'FAILED'}",
    "=" * 65
])

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n".join(report))

if failed_count > 0:
    raise ValueError(
        "Final validation failed. Review the failed checks."
    )

print("\nSTEP 9E COMPLETED SUCCESSFULLY")
