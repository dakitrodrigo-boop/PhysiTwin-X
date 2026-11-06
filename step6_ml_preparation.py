
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

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. APPROVED PREDICTOR VARIABLES
# ============================================================

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

TARGET = "remaining_shelf_life_days"
GROUP = "mango_id"

# Variables that must NOT be used as predictors
EXCLUDED_REASONS = {
    "endpoint_day": "Future endpoint information; target leakage",
    "ripeness_stage": "Excluded to prevent spoilage-label leakage",
    "spoilage_risk": "Spoilage classification label; not a predictor",
    "mango_id": "Grouping identifier; not a predictor",
    "remaining_shelf_life_days": "Target variable; not a predictor",
    "review_flag": "Data review metadata; not a predictor"
}

# ============================================================
# 2. LOAD VERIFIED DATASET
# ============================================================

print("=" * 65)
print("PHYSITWIN-X STEP 6: MACHINE LEARNING DATA PREPARATION")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nINPUT DATASET")
print("File:", INPUT_FILE.name)
print("Observations:", len(df))
print("Unique mangoes:", df[GROUP].nunique())

# ============================================================
# 3. DATA VALIDATION
# ============================================================

required_columns = FEATURES + [TARGET, GROUP]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[GROUP] = df[GROUP].astype(str).str.strip()

if df[GROUP].eq("").any():
    raise ValueError("Missing or empty mango IDs detected.")

if df.duplicated(subset=[GROUP, "storage_day"]).any():
    raise ValueError("Duplicate mango and storage-day records detected.")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

if df[TARGET].isna().any():
    raise ValueError("Missing or invalid shelf-life targets detected.")

if (df[TARGET] < 0).any():
    raise ValueError("Negative shelf-life targets detected.")

if df[GROUP].nunique() != 60:
    raise ValueError(
        f"Expected 60 mangoes, found {df[GROUP].nunique()}."
    )

if not df["storage_day"].between(1, 14).all():
    raise ValueError("Storage days must be within Day 1 to Day 14.")

print("\nDATA VALIDATION")
print("Missing required columns: 0")
print("Missing targets:", df[TARGET].isna().sum())
print("Duplicate mango/day records:", 
      df.duplicated([GROUP, "storage_day"]).sum())
print("Negative targets:", (df[TARGET] < 0).sum())
print("Observation period: Day 1 to Day 14")
print("Validation: PASSED")

# ============================================================
# 4. FEATURE AUDIT
# ============================================================

audit_rows = []

for col in df.columns:

    if col in FEATURES:
        role = "Predictor"
        reason = "Approved current or historical measurement"

    elif col == TARGET:
        role = "Target"
        reason = "Remaining shelf-life prediction target"

    elif col == GROUP:
        role = "Group identifier"
        reason = "Used only for mango-level data splitting"

    else:
        role = "Excluded"
        reason = EXCLUDED_REASONS.get(
            col,
            "Not included in the approved predictor list"
        )

    audit_rows.append({
        "variable": col,
        "role": role,
        "data_type": str(df[col].dtype),
        "missing_values": int(df[col].isna().sum()),
        "missing_percentage": round(
            df[col].isna().mean() * 100, 2
        ),
        "unique_values": int(df[col].nunique(dropna=True)),
        "reason": reason
    })

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_FILE, index=False)

print("\nFEATURE AUDIT")
print("Approved predictors:", len(FEATURES))
print("Target variable:", TARGET)
print("Grouping variable:", GROUP)

# ============================================================
# 5. PREPARE MODELING DATA
# ============================================================

selected_columns = [GROUP] + FEATURES + [TARGET]

model_data = df[selected_columns].copy()

# Keep categorical values as text.
# Missing predictor values are retained for later
# preprocessing inside the machine-learning pipeline.

categorical_features = [
    "peel_color",
    "firmness"
]

for col in categorical_features:
    model_data[col] = model_data[col].astype("string")

# ============================================================
# 6. MANGO-LEVEL TRAIN/TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        model_data[FEATURES],
        model_data[TARGET],
        groups=model_data[GROUP]
    )
)

train_data = model_data.iloc[train_indices].copy()
test_data = model_data.iloc[test_indices].copy()

# ============================================================
# 7. VERIFY GROUP SEPARATION
# ============================================================

train_mangoes = set(train_data[GROUP].unique())
test_mangoes = set(test_data[GROUP].unique())

overlap = train_mangoes.intersection(test_mangoes)

if overlap:
    raise ValueError(
        f"Data leakage detected! Shared mangoes: {overlap}"
    )

if len(train_mangoes) + len(test_mangoes) != 60:
    raise ValueError("Unexpected number of mangoes after splitting.")

if len(train_data) + len(test_data) != len(model_data):
    raise ValueError("Some observations were lost during splitting.")

# ============================================================
# 8. SAVE TRAINING AND TESTING DATASETS
# ============================================================

train_data.to_csv(TRAIN_FILE, index=False)
test_data.to_csv(TEST_FILE, index=False)

# ============================================================
# 9. GENERATE SUMMARY REPORT
# ============================================================

summary = []

summary.append("=" * 65)
summary.append("PHYSITWIN-X STEP 6 SUMMARY REPORT")
summary.append("=" * 65)

summary.append("")
summary.append("SOURCE DATASET")
summary.append(f"Total observations: {len(df)}")
summary.append(f"Total mangoes: {df[GROUP].nunique()}")
summary.append("Observation period: Day 1 to Day 14")

summary.append("")
summary.append("MACHINE LEARNING VARIABLES")
summary.append(f"Number of predictors: {len(FEATURES)}")
summary.append(f"Target variable: {TARGET}")
summary.append(f"Grouping variable: {GROUP}")

summary.append("")
summary.append("TRAINING DATASET")
summary.append(f"Observations: {len(train_data)}")
summary.append(f"Mangoes: {len(train_mangoes)}")

summary.append("")
summary.append("TESTING DATASET")
summary.append(f"Observations: {len(test_data)}")
summary.append(f"Mangoes: {len(test_mangoes)}")

summary.append("")
summary.append("GROUP SEPARATION VALIDATION")
summary.append(f"Shared mango IDs: {len(overlap)}")
summary.append("Data leakage check: PASSED")

summary.append("")
summary.append("MISSING PREDICTOR VALUES")

for col in FEATURES:
    count = int(model_data[col].isna().sum())
    summary.append(f"{col}: {count}")

summary.append("")
summary.append("OUTPUT FILES")
summary.append(str(TRAIN_FILE))
summary.append(str(TEST_FILE))
summary.append(str(AUDIT_FILE))

summary.append("")
summary.append("STEP 6 STATUS: COMPLETED SUCCESSFULLY")

SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8"
)

# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TRAINING AND TESTING SPLIT")
print("=" * 65)

print("\nTRAINING DATA")
print("Observations:", len(train_data))
print("Mangoes:", len(train_mangoes))

print("\nTESTING DATA")
print("Observations:", len(test_data))
print("Mangoes:", len(test_mangoes))

print("\nGROUP SEPARATION")
print("Shared mango IDs:", len(overlap))
print("Data leakage check: PASSED")

print("\nOUTPUT FILES")
print("Training:", TRAIN_FILE)
print("Testing:", TEST_FILE)
print("Feature audit:", AUDIT_FILE)
print("Summary:", SUMMARY_FILE)

print("\n" + "=" * 65)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 65)
