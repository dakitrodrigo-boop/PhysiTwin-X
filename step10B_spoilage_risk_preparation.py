
from pathlib import Path
import pandas as pd

# ============================================================
# PHYSITWIN-X STEP 10B
# REVISED SPOILAGE-RISK DATA PREPARATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

SOURCE_FILE = DATA_DIR / "physitwin_x_endpoint_reverified.csv"
TRAIN_FILE = DATA_DIR / "physitwin_x_train.csv"
TEST_FILE = DATA_DIR / "physitwin_x_test.csv"

TRAIN_OUTPUT = DATA_DIR / "physitwin_x_risk_train.csv"
TEST_OUTPUT = DATA_DIR / "physitwin_x_risk_test.csv"

REPORT_FILE = RESULTS_DIR / "step10B_risk_preparation_report.txt"

TARGET = "spoilage_risk"

FEATURES = [
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "peel_color",
    "firmness",
    "days_since_first_observation",
    "observation_gap_days",
    "weight_change_g",
    "weight_loss_change_pct",
    "weight_loss_rate_pct_per_day",
    "temperature_change_c",
    "humidity_change_pct"
]

EXPECTED_MISSING_COLUMNS = [
    "observation_gap_days",
    "weight_change_g",
    "weight_loss_change_pct",
    "weight_loss_rate_pct_per_day",
    "temperature_change_c",
    "humidity_change_pct"
]

lines = []
checks = []

def log(message=""):
    print(message)
    lines.append(str(message))

def check(description, condition):
    passed = bool(condition)
    checks.append((description, passed))
    status = "PASS" if passed else "FAIL"
    log(f"[{status}] {description}")

# ============================================================
# LOAD DATA
# ============================================================

log("=" * 65)
log("PHYSITWIN-X STEP 10B")
log("REVISED SPOILAGE-RISK DATA PREPARATION")
log("=" * 65)

source = pd.read_csv(SOURCE_FILE)
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# Check duplicate column names
check(
    "No duplicate source column names",
    source.columns.is_unique
)

check(
    "No duplicate training column names",
    train.columns.is_unique
)

check(
    "No duplicate testing column names",
    test.columns.is_unique
)

# Check required columns
required = [
    "mango_id",
    "storage_day",
    TARGET
] + FEATURES

check(
    "All required source columns exist",
    set(required).issubset(source.columns)
)

check(
    "All required training columns exist",
    set(["mango_id"] + FEATURES).issubset(train.columns)
)

check(
    "All required testing columns exist",
    set(["mango_id"] + FEATURES).issubset(test.columns)
)

if not all(result for _, result in checks[-3:]):
    raise ValueError("Required columns are missing. Review the dataset.")

# ============================================================
# PREPARE RISK LABELS
# ============================================================

risk_labels = source[
    ["mango_id", "storage_day", TARGET]
].copy()

check(
    "No duplicate mango/day records in source",
    not risk_labels.duplicated(
        ["mango_id", "storage_day"]
    ).any()
)

check(
    "No missing spoilage-risk labels in source",
    risk_labels[TARGET].notna().all()
)

# ============================================================
# MERGE LABELS INTO TRAINING AND TESTING DATA
# ============================================================

def prepare_split(split, split_name):

    selected = split[["mango_id"] + FEATURES].copy()

    prepared = selected.merge(
        risk_labels,
        on=["mango_id", "storage_day"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    merge_success = (prepared["_merge"] == "both").all()

    prepared = prepared.drop(columns=["_merge"])

    check(
        f"{split_name}: all records matched with risk labels",
        merge_success
    )

    return prepared


risk_train = prepare_split(train, "TRAINING")
risk_test = prepare_split(test, "TESTING")

# ============================================================
# EXPECTED MISSING-VALUE VALIDATION
# ============================================================

def validate_missing_values(df, split_name):

    ordered = df.sort_values(
        ["mango_id", "storage_day"]
    ).copy()

    first_rows = (
        ordered.groupby("mango_id", sort=False)
        .head(1)
        .index
    )

    first_mask = ordered.index.isin(first_rows)

    missing_counts = ordered.isna().sum()

    actual_missing_columns = set(
        missing_counts[missing_counts > 0].index
    )

    check(
        f"{split_name}: missing values limited to six expected columns",
        actual_missing_columns.issubset(
            set(EXPECTED_MISSING_COLUMNS)
        )
    )

    # Expected missing values must occur only in first observations.
    missing_only_first = True

    for column in EXPECTED_MISSING_COLUMNS:

        missing_mask = ordered[column].isna()

        if (missing_mask & ~first_mask).any():
            missing_only_first = False

    check(
        f"{split_name}: expected missing values occur only in first observations",
        missing_only_first
    )

    # Every mango should have missing lag values at its first observation.
    first_missing_correct = True

    for column in EXPECTED_MISSING_COLUMNS:

        if not ordered.loc[first_mask, column].isna().all():
            first_missing_correct = False

    check(
        f"{split_name}: all first observations have unavailable lag values",
        first_missing_correct
    )

    # All other predictors and the target must be complete.
    other_columns = [
        column for column in FEATURES + [TARGET]
        if column not in EXPECTED_MISSING_COLUMNS
    ]

    no_other_missing = (
        not ordered[other_columns].isna().any().any()
    )

    check(
        f"{split_name}: all other predictors and labels are complete",
        no_other_missing
    )

    log(f"\n{split_name} missing-value summary:")

    log(
        f"Total missing values: "
        f"{int(ordered.isna().sum().sum())}"
    )

    log(
        f"Expected missing values: "
        f"{int(ordered[EXPECTED_MISSING_COLUMNS].isna().sum().sum())}"
    )

    log(
        f"Missing values outside expected columns: "
        f"{int(ordered.drop(columns=EXPECTED_MISSING_COLUMNS).isna().sum().sum())}"
    )


validate_missing_values(risk_train, "TRAINING")
validate_missing_values(risk_test, "TESTING")

# ============================================================
# GROUPED TRAIN/TEST VALIDATION
# ============================================================

train_ids = set(risk_train["mango_id"].unique())
test_ids = set(risk_test["mango_id"].unique())
source_ids = set(source["mango_id"].unique())

check(
    "Training and testing mangoes do not overlap",
    train_ids.isdisjoint(test_ids)
)

check(
    "All source mangoes are represented",
    train_ids.union(test_ids) == source_ids
)

check(
    "Training contains 48 mangoes",
    len(train_ids) == 48
)

check(
    "Testing contains 12 mangoes",
    len(test_ids) == 12
)

# ============================================================
# RISK CLASS VALIDATION
# ============================================================

expected_classes = {
    "FRESH",
    "LOW RISK",
    "MODERATE RISK",
    "HIGH RISK",
    "VERY HIGH RISK"
}

train_classes = set(risk_train[TARGET].dropna().unique())
test_classes = set(risk_test[TARGET].dropna().unique())

check(
    "Training contains all five risk classes",
    train_classes == expected_classes
)

check(
    "Testing contains all five risk classes",
    test_classes == expected_classes
)

# ============================================================
# SAVE DATASETS
# ============================================================

risk_train.to_csv(TRAIN_OUTPUT, index=False)
risk_test.to_csv(TEST_OUTPUT, index=False)

# ============================================================
# REPORT
# ============================================================

log("\n" + "=" * 65)
log("DATASET SUMMARY")
log("=" * 65)

log(f"Source observations: {len(source)}")
log(f"Source mangoes: {source['mango_id'].nunique()}")

log(f"Training observations: {len(risk_train)}")
log(f"Training mangoes: {risk_train['mango_id'].nunique()}")

log(f"Testing observations: {len(risk_test)}")
log(f"Testing mangoes: {risk_test['mango_id'].nunique()}")

log("\nTRAINING RISK DISTRIBUTION")
log(risk_train[TARGET].value_counts().to_string())

log("\nTESTING RISK DISTRIBUTION")
log(risk_test[TARGET].value_counts().to_string())

log("\n" + "=" * 65)
log("VALIDATION RESULTS")
log("=" * 65)

passed = sum(result for _, result in checks)
failed = len(checks) - passed

log(f"Checks passed: {passed}/{len(checks)}")
log(f"Checks failed: {failed}")

if failed == 0:

    log("\nSTEP 10B PASSED")
    log("Risk datasets are ready for classification modeling.")

else:

    log("\nSTEP 10B NEEDS REVIEW")
    log("Review failed checks before proceeding.")

log("\nOUTPUT FILES")
log(str(TRAIN_OUTPUT))
log(str(TEST_OUTPUT))

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print(f"\nReport saved to: {REPORT_FILE}")

if failed > 0:
    raise SystemExit("STEP 10B validation failed.")
