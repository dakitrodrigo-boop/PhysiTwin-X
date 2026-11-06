
from pathlib import Path
import pandas as pd

# ============================================================
# PHYSITWIN-X STEP 10A: SPOILAGE-RISK DATA AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

SOURCE_FILE = DATA_DIR / "physitwin_x_endpoint_reverified.csv"
TRAIN_FILE = DATA_DIR / "physitwin_x_train.csv"
TEST_FILE = DATA_DIR / "physitwin_x_test.csv"

REPORT_FILE = RESULTS_DIR / "step10A_spoilage_risk_audit.txt"
CLASS_FILE = RESULTS_DIR / "step10A_risk_class_distribution.csv"
MANGO_FILE = RESULTS_DIR / "step10A_individual_mango_risk_summary.csv"

EXPECTED_CLASSES = [
    "FRESH",
    "LOW RISK",
    "MODERATE RISK",
    "HIGH RISK",
    "VERY HIGH RISK"
]

print("=" * 65)
print("PHYSITWIN-X STEP 10A: SPOILAGE-RISK DATA AUDIT")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(SOURCE_FILE)
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

required = ["mango_id", "storage_day", "spoilage_risk"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required source columns: {missing}")

# Standardize risk labels for inspection
df["spoilage_risk"] = (
    df["spoilage_risk"]
    .astype("string")
    .str.strip()
    .str.upper()
)

# ------------------------------------------------------------
# 2. BASIC DATA VALIDATION
# ------------------------------------------------------------

checks = []

def check(description, condition):
    status = bool(condition)
    checks.append((description, status))
    print(("PASS: " if status else "FAIL: ") + description)

check("Source file loaded", len(df) > 0)
check("Mango IDs are not missing", df["mango_id"].notna().all())
check("Storage days are not missing", df["storage_day"].notna().all())
check("Spoilage-risk labels are not missing",
      df["spoilage_risk"].notna().all())
check("No duplicate mango-day records",
      not df.duplicated(["mango_id", "storage_day"]).any())
check("Exactly 60 unique mangoes",
      df["mango_id"].nunique() == 60)

# ------------------------------------------------------------
# 3. RISK CLASS DISTRIBUTION
# ------------------------------------------------------------

class_counts = (
    df["spoilage_risk"]
    .value_counts(dropna=False)
    .rename_axis("spoilage_risk")
    .reset_index(name="observations")
)

class_counts["percentage"] = (
    class_counts["observations"] / len(df) * 100
)

class_counts.to_csv(CLASS_FILE, index=False)

observed_classes = set(df["spoilage_risk"].dropna().unique())
expected_classes = set(EXPECTED_CLASSES)

check("All five expected risk classes are present",
      observed_classes == expected_classes)

print("\nRISK CLASS DISTRIBUTION")
print(class_counts.to_string(index=False))

# ------------------------------------------------------------
# 4. INDIVIDUAL MANGO RISK SUMMARY
# ------------------------------------------------------------

mango_summary = (
    df.groupby("mango_id")
    .agg(
        observations=("storage_day", "count"),
        first_day=("storage_day", "min"),
        last_day=("storage_day", "max"),
        number_of_risk_classes=("spoilage_risk", "nunique"),
        first_recorded_risk=("spoilage_risk", "first"),
        last_recorded_risk=("spoilage_risk", "last")
    )
    .reset_index()
)

mango_summary.to_csv(MANGO_FILE, index=False)

print("\nINDIVIDUAL MANGO RISK SUMMARY")
print(mango_summary.to_string(index=False))

# ------------------------------------------------------------
# 5. CHECK THE EXISTING MANGO-LEVEL TRAIN/TEST SPLIT
# ------------------------------------------------------------

if "mango_id" not in train.columns or "mango_id" not in test.columns:
    raise ValueError("Train/test files must contain mango_id.")

train_ids = set(train["mango_id"].dropna().unique())
test_ids = set(test["mango_id"].dropna().unique())
source_ids = set(df["mango_id"].dropna().unique())

overlap = train_ids.intersection(test_ids)
missing_split_ids = source_ids - (train_ids | test_ids)
unknown_split_ids = (train_ids | test_ids) - source_ids

check("No mango appears in both train and test sets",
      len(overlap) == 0)

check("All source mangoes are assigned to train or test",
      len(missing_split_ids) == 0)

check("No unknown mango IDs in train/test files",
      len(unknown_split_ids) == 0)

print("\nEXISTING GROUPED SPLIT")
print(f"Training mangoes: {len(train_ids)}")
print(f"Testing mangoes: {len(test_ids)}")
print(f"Overlapping mangoes: {len(overlap)}")

# ------------------------------------------------------------
# 6. RISK LABELS ACROSS STORAGE DAYS
# ------------------------------------------------------------

daily_risk = pd.crosstab(
    df["storage_day"],
    df["spoilage_risk"]
)

print("\nRISK LABELS BY STORAGE DAY")
print(daily_risk.to_string())

# ------------------------------------------------------------
# 7. FINAL REPORT
# ------------------------------------------------------------

passed = sum(status for _, status in checks)
failed = len(checks) - passed

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("PHYSITWIN-X STEP 10A: SPOILAGE-RISK AUDIT\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Total observations: {len(df)}\n")
    file.write(f"Unique mangoes: {df['mango_id'].nunique()}\n")
    file.write(
        f"Storage days: {df['storage_day'].min()} "
        f"to {df['storage_day'].max()}\n\n"
    )

    file.write("VALIDATION RESULTS\n")
    for description, status in checks:
        file.write(
            f"{'PASS' if status else 'FAIL'}: {description}\n"
        )

    file.write(f"\nPassed: {passed}\n")
    file.write(f"Failed: {failed}\n")

    file.write("\nRISK CLASS DISTRIBUTION\n")
    file.write(class_counts.to_string(index=False))

    file.write("\n\nEXISTING GROUPED SPLIT\n")
    file.write(f"Training mangoes: {len(train_ids)}\n")
    file.write(f"Testing mangoes: {len(test_ids)}\n")
    file.write(f"Overlapping mangoes: {len(overlap)}\n")

    file.write("\n\nRISK LABELS BY STORAGE DAY\n")
    file.write(daily_risk.to_string())

    file.write("\n\nIMPORTANT NOTE\n")
    file.write(
        "This step audits recorded risk labels. It does not train "
        "or validate a spoilage-risk prediction model.\n"
    )

print("\n" + "=" * 65)
print("STEP 10A AUDIT SUMMARY")
print("=" * 65)
print(f"Total checks: {len(checks)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

print("\nOUTPUT FILES")
print(CLASS_FILE)
print(MANGO_FILE)
print(REPORT_FILE)

if failed == 0:
    print("\nSTEP 10A AUDIT COMPLETED: ALL CHECKS PASSED")
else:
    print("\nSTEP 10A AUDIT NEEDS REVIEW")
