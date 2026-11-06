
from pathlib import Path
import pandas as pd

# ============================================================
# PHYSITWIN-X STEP 10B-1
# MISSING-VALUE DIAGNOSTIC
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

FILES = [
    "physitwin_x_risk_train.csv",
    "physitwin_x_risk_test.csv"
]

REPORT = RESULTS_DIR / "step10B1_missing_value_report.txt"

lines = []

def log(message=""):
    print(message)
    lines.append(str(message))

log("=" * 65)
log("PHYSITWIN-X STEP 10B-1: MISSING-VALUE DIAGNOSTIC")
log("=" * 65)

for filename in FILES:

    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    df = pd.read_csv(filepath)

    log(f"\nDATASET: {filename}")
    log("-" * 65)

    log(f"Observations: {len(df)}")
    log(f"Columns: {len(df.columns)}")
    log(f"Unique mangoes: {df['mango_id'].nunique()}")

    missing = df.isna().sum()
    missing_columns = missing[missing > 0]

    log(f"\nTotal missing values: {int(missing.sum())}")

    if missing_columns.empty:
        log("No missing values found.")
    else:
        log("\nMISSING VALUES BY COLUMN")
        log(missing_columns.to_string())

    # Inspect first observation for each mango
    first_records = (
        df.sort_values(["mango_id", "storage_day"])
        .groupby("mango_id")
        .head(1)
    )

    log("\nMISSING VALUES IN FIRST OBSERVATION")
    first_missing = first_records.isna().sum()
    first_missing = first_missing[first_missing > 0]

    if first_missing.empty:
        log("No missing values in first observations.")
    else:
        log(first_missing.to_string())

    # Inspect missing values outside first observations
    remaining_records = (
        df.sort_values(["mango_id", "storage_day"])
        .groupby("mango_id")
        .apply(lambda group: group.iloc[1:])
        .reset_index(drop=True)
    )

    log("\nMISSING VALUES AFTER FIRST OBSERVATION")

    later_missing = remaining_records.isna().sum()
    later_missing = later_missing[later_missing > 0]

    if later_missing.empty:
        log("No missing values after first observations.")
    else:
        log(later_missing.to_string())

    # Show example rows containing missing values
    missing_rows = df[df.isna().any(axis=1)]

    log("\nNUMBER OF ROWS WITH MISSING VALUES")
    log(str(len(missing_rows)))

    if not missing_rows.empty:
        log("\nFIRST 10 AFFECTED RECORDS")
        log(missing_rows.head(10).to_string(index=False))

    log("\n" + "=" * 65)

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\nDIAGNOSTIC COMPLETED")
print(f"Report saved to: {REPORT}")
