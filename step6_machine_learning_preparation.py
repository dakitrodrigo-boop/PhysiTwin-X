
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# PHYSITWIN-X
# STEP 6: MACHINE LEARNING DATA PREPARATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = DATA_DIR / "physitwin_x_shelf_life_targets.csv"

TRAIN_FILE = DATA_DIR / "physitwin_x_train.csv"
TEST_FILE = DATA_DIR / "physitwin_x_test.csv"
AUDIT_FILE = RESULTS_DIR / "step6_feature_audit.csv"
SUMMARY_FILE = RESULTS_DIR / "step6_split_summary.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "remaining_shelf_life_days"
GROUP = "mango_id"

# Only information available at the current observation time.
# Ripeness and spoilage labels are excluded from this baseline
# because they may reveal information closely related to the target.
CANDIDATE_FEATURES = [
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
    "humidity_change_pct",
]

# ============================================================
# 1. CHECK FILES AND FOLDERS
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Check that Step 5 completed and the CSV is inside the data folder."
    )

print("=" * 65)
print("PHYSITWIN-X: STEP 6 MACHINE LEARNING PREPARATION")
print("=" * 65)

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Standardize column names only in memory.
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("\nINPUT DATASET")
print("File:", INPUT_FILE.name)
print("Observations:", len(df))
print("Columns:", len(df.columns))

# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [GROUP, TARGET]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Check duplicate mango/day observations.
if "storage_day" in df.columns:
    duplicate_count = df.duplicated(
        subset=[GROUP, "storage_day"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate mango/day records."
        )

# ============================================================
# 4. CHECK TRAINING ELIGIBILITY
# ============================================================

if "shelf_life_training_eligible" in df.columns:
    eligible = (
        df["shelf_life_training_eligible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    excluded_count = int((~eligible).sum())
    df = df.loc[eligible].copy()

    print("\nEligibility filter applied.")
    print("Excluded observations:", excluded_count)
else:
    print("\nNo eligibility column found.")
    print("Proceeding with required target and review checks.")

# Exclude missing targets and unresolved review records.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[GROUP, TARGET]).copy()

if "data_review_required" in df.columns:
    review_flags = (
        df["data_review_required"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    df = df.loc[~review_flags].copy()

if "post_endpoint_observation" in df.columns:
    post_endpoint = (
        df["post_endpoint_observation"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    df = df.loc[~post_endpoint].copy()

if (df[TARGET] < 0).any():
    raise ValueError("Negative shelf-life targets were found.")

# ============================================================
# 5. SELECT PREDICTION FEATURES
# ============================================================

selected_features = [
    col for col in CANDIDATE_FEATURES
    if col in df.columns
]

missing_features = [
    col for col in CANDIDATE_FEATURES
    if col not in df.columns
]

if not selected_features:
    raise ValueError(
        "No candidate prediction features were found. "
        "Check the dataset column names."
    )

# Remove features that are entirely empty.
selected_features = [
    col for col in selected_features
    if not df[col].isna().all()
]

if not selected_features:
    raise ValueError("All selected features are empty.")

# Confirm that forbidden leakage columns are not predictors.
forbidden_features = {
    TARGET,
    "endpoint_day",
    "spoilage_risk",
    "ripeness_stage",
    "mango_id",
    "post_endpoint_observation",
    "shelf_life_training_eligible",
    "data_review_required",
}

selected_features = [
    col for col in selected_features
    if col not in forbidden_features
]

print("\nSELECTED PREDICTION FEATURES")
for feature in selected_features:
    print("-", feature)

print("\nCANDIDATE FEATURES NOT FOUND")
for feature in missing_features:
    print("-", feature)

# ============================================================
# 6. PREPARE MODELING DATA
# ============================================================

model_data = df[
    [GROUP] + selected_features + [TARGET]
].copy()

# Check data types and missing feature values.
feature_types = model_data[selected_features].dtypes.astype(str)

feature_missing = model_data[selected_features].isna().sum()

audit = pd.DataFrame({
    "feature": selected_features,
    "data_type": [
        feature_types[col] for col in selected_features
    ],
    "missing_values": [
        int(feature_missing[col]) for col in selected_features
    ],
    "selected_as_predictor": True,
})

audit.to_csv(AUDIT_FILE, index=False)

# ============================================================
# 7. SPLIT BY MANGO ID
# ============================================================

groups = model_data[GROUP]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(
        model_data,
        model_data[TARGET],
        groups=groups,
    )
)

train_data = model_data.iloc[train_indices].copy()
test_data = model_data.iloc[test_indices].copy()

train_mangoes = set(train_data[GROUP].unique())
test_mangoes = set(test_data[GROUP].unique())

# ============================================================
# 8. VERIFY NO MANGO-ID LEAKAGE
# ============================================================

overlap = train_mangoes.intersection(test_mangoes)

if overlap:
    raise ValueError(
        f"Mango ID leakage detected: {sorted(overlap)}"
    )

if len(train_mangoes) + len(test_mangoes) != model_data[GROUP].nunique():
    raise ValueError("Some mango IDs were not included in the split.")

# ============================================================
# 9. SAVE TRAINING AND TESTING DATA
# ============================================================

train_data.to_csv(TRAIN_FILE, index=False)
test_data.to_csv(TEST_FILE, index=False)

# ============================================================
# 10. SAVE SPLIT SUMMARY
# ============================================================

summary = f"""
PHYSITWIN-X - STEP 6 SPLIT SUMMARY

INPUT FILE:
{INPUT_FILE.name}

TOTAL ELIGIBLE OBSERVATIONS:
{len(model_data)}

TOTAL MANGOES:
{model_data[GROUP].nunique()}

TRAINING SET:
{len(train_data)} observations
{len(train_mangoes)} mangoes

TESTING SET:
{len(test_data)} observations
{len(test_mangoes)} mangoes

TRAINING MANGO IDS:
{sorted(train_mangoes)}

TESTING MANGO IDS:
{sorted(test_mangoes)}

PREDICTION TARGET:
{TARGET}

SELECTED FEATURES:
{selected_features}

FEATURES NOT FOUND:
{missing_features}

MANGO-ID OVERLAP:
{len(overlap)}

LEAKAGE CHECK:
PASSED - No mango ID appears in both sets.

IMPORTANT:
The endpoint day, target column, spoilage-risk label,
and ripeness-stage label were not used as predictors.

This step prepares data only. It does not train a model.
"""

SUMMARY_FILE.write_text(summary, encoding="utf-8")

# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("STEP 6 RESULTS")
print("=" * 65)

print("Eligible observations:", len(model_data))
print("Total mangoes:", model_data[GROUP].nunique())

print("\nTRAINING SET")
print("Observations:", len(train_data))
print("Mangoes:", len(train_mangoes))

print("\nTESTING SET")
print("Observations:", len(test_data))
print("Mangoes:", len(test_mangoes))

print("\nMANGO ID OVERLAP:", len(overlap))
print("LEAKAGE CHECK: PASSED")

print("\nSAVED FILES")
print(TRAIN_FILE)
print(TEST_FILE)
print(AUDIT_FILE)
print(SUMMARY_FILE)

print("\nSTEP 6 COMPLETED SUCCESSFULLY.")
