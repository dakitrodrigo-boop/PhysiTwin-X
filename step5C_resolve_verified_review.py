
# ================================================================
# PHYSITWIN-X
# STEP 5: SHELF-LIFE TARGET GENERATION
# ================================================================
#
# Purpose:
# 1. Identify the first SPOILED observation for each mango.
# 2. Calculate remaining shelf life in days.
# 3. Identify observations recorded after the spoilage endpoint.
# 4. Exclude post-endpoint observations from shelf-life training.
# 5. Preserve all original observations.
# 6. Generate an endpoint audit report.
#
# Input:
# data/physitwin_x_review_verified.csv
#
# Outputs:
# data/physitwin_x_shelf_life_targets.csv
# results/shelf_life_endpoint_audit.csv
#
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ================================================================
# 1. FILE PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = DATA_DIR / "physitwin_x_review_verified.csv"

OUTPUT_FILE = DATA_DIR / "physitwin_x_shelf_life_targets.csv"

AUDIT_FILE = RESULTS_DIR / "shelf_life_endpoint_audit.csv"


# Create results directory if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. LOAD VERIFIED DATASET
# ================================================================

print("=" * 65)
print("PHYSITWIN-X STEP 5: SHELF-LIFE TARGET GENERATION")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDATASET INFORMATION")
print("-" * 65)
print(f"Input file: {INPUT_FILE.name}")
print(f"Total observations: {len(df)}")
print(f"Total mangoes: {df['mango_id'].nunique()}")


# ================================================================
# 3. CHECK REQUIRED COLUMNS
# ================================================================

required_columns = [
    "mango_id",
    "storage_day",
    "ripeness_stage"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Check for missing identification information
if df["mango_id"].isna().any():
    raise ValueError("Missing mango_id values detected.")

if df["storage_day"].isna().any():
    raise ValueError("Missing storage_day values detected.")

# Convert storage day to numeric
df["storage_day"] = pd.to_numeric(
    df["storage_day"],
    errors="coerce"
)

if df["storage_day"].isna().any():
    raise ValueError("Invalid storage_day values detected.")


# ================================================================
# 4. IDENTIFY FIRST SPOILED DAY FOR EACH MANGO
# ================================================================

df["ripeness_stage"] = (
    df["ripeness_stage"]
    .astype("string")
    .str.strip()
    .str.upper()
)

spoiled_records = df[
    df["ripeness_stage"] == "SPOILED"
].copy()

endpoint_days = (
    spoiled_records
    .groupby("mango_id")["storage_day"]
    .min()
)

df["endpoint_day"] = df["mango_id"].map(endpoint_days)


# ================================================================
# 5. VALIDATE SPOILAGE ENDPOINTS
# ================================================================

total_mangoes = df["mango_id"].nunique()

mangoes_with_endpoints = endpoint_days.index.nunique()

mangoes_without_endpoints = (
    total_mangoes - mangoes_with_endpoints
)

if mangoes_without_endpoints > 0:

    missing_endpoint_ids = sorted(
        set(df["mango_id"].unique())
        - set(endpoint_days.index)
    )

    raise ValueError(
        "Some mangoes have no SPOILED observation.\n"
        f"Mangoes without endpoints: {missing_endpoint_ids}\n"
        "Verify the original observations before proceeding."
    )

# Check that the endpoint is not earlier than the first
# recorded observation for a mango.
first_days = df.groupby("mango_id")["storage_day"].min()

invalid_endpoints = endpoint_days[
    endpoint_days < first_days.reindex(endpoint_days.index)
]

if not invalid_endpoints.empty:
    raise ValueError(
        "Invalid spoilage endpoint detected:\n"
        f"{invalid_endpoints}"
    )


# ================================================================
# 6. CALCULATE REMAINING SHELF LIFE
# ================================================================

df["remaining_shelf_life_days"] = (
    df["endpoint_day"] - df["storage_day"]
)

# Identify observations recorded after the first SPOILED day
df["post_endpoint_observation"] = (
    df["storage_day"] > df["endpoint_day"]
)

# Post-endpoint observations must not have negative targets.
df.loc[
    df["post_endpoint_observation"],
    "remaining_shelf_life_days"
] = np.nan


# ================================================================
# 7. CHECK AND HANDLE DATA-REVIEW FLAGS
# ================================================================

if "data_review_required" not in df.columns:
    df["data_review_required"] = False

# Treat missing review flags as requiring review.
df["data_review_required"] = (
    df["data_review_required"]
    .fillna(True)
    .astype(bool)
)

# Only observations with valid targets and no review flags
# can be used for shelf-life model training.
df["shelf_life_training_eligible"] = (
    df["remaining_shelf_life_days"].notna()
    & (~df["data_review_required"])
    & (~df["post_endpoint_observation"])
)


# ================================================================
# 8. GENERATE MANGO ENDPOINT AUDIT
# ================================================================

audit_rows = []

for mango_id, group in df.groupby("mango_id"):

    endpoint = endpoint_days.loc[mango_id]

    first_observation = group["storage_day"].min()
    last_observation = group["storage_day"].max()

    spoiled_count = (
        group["ripeness_stage"] == "SPOILED"
    ).sum()

    audit_rows.append({
        "mango_id": mango_id,
        "first_observation_day": first_observation,
        "last_observation_day": last_observation,
        "endpoint_day": endpoint,
        "total_observations": len(group),
        "spoiled_observations": spoiled_count,
        "post_endpoint_observations": int(
            group["post_endpoint_observation"].sum()
        ),
        "eligible_training_observations": int(
            group["shelf_life_training_eligible"].sum()
        )
    })

audit_df = pd.DataFrame(audit_rows)

audit_df = audit_df.sort_values("mango_id")


# ================================================================
# 9. CALCULATE VALIDATION SUMMARY
# ================================================================

total_observations = len(df)

eligible_observations = int(
    df["shelf_life_training_eligible"].sum()
)

review_flags = int(
    df["data_review_required"].sum()
)

post_endpoint_count = int(
    df["post_endpoint_observation"].sum()
)

blank_post_endpoint_targets = int(
    df.loc[
        df["post_endpoint_observation"],
        "remaining_shelf_life_days"
    ].isna().sum()
)

negative_targets = int(
    (df["remaining_shelf_life_days"] < 0).sum()
)

missing_targets = int(
    df["remaining_shelf_life_days"].isna().sum()
)

# Verify that all post-endpoint targets are blank
if blank_post_endpoint_targets != post_endpoint_count:
    raise ValueError(
        "Post-endpoint target validation failed."
    )

# Verify that there are no negative remaining shelf-life targets
if negative_targets != 0:
    raise ValueError(
        f"Negative targets detected: {negative_targets}"
    )

# Verify that all mangoes have a spoilage endpoint
if mangoes_with_endpoints != total_mangoes:
    raise ValueError(
        "Not all mangoes have verified spoilage endpoints."
    )


# ================================================================
# 10. SAVE OUTPUT FILES
# ================================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

audit_df.to_csv(
    AUDIT_FILE,
    index=False
)


# ================================================================
# 11. DISPLAY FINAL REPORT
# ================================================================

print("\nSHELF-LIFE TARGET SUMMARY")
print("-" * 65)

print(f"Total observations: {total_observations}")
print(f"Total mangoes: {total_mangoes}")

print(
    f"Mangoes with SPOILED endpoints: "
    f"{mangoes_with_endpoints}"
)

print(
    f"Mangoes without endpoints: "
    f"{mangoes_without_endpoints}"
)

print(
    f"Eligible shelf-life training observations: "
    f"{eligible_observations}"
)

print(
    f"Records requiring data review: "
    f"{review_flags}"
)

print(
    f"Post-endpoint observations: "
    f"{post_endpoint_count}"
)

print(
    f"Blank post-endpoint targets: "
    f"{blank_post_endpoint_targets}"
)

print(
    f"Negative shelf-life targets: "
    f"{negative_targets}"
)

print(
    f"All missing shelf-life targets: "
    f"{missing_targets}"
)


# ================================================================
# 12. DISPLAY POST-ENDPOINT RECORDS
# ================================================================

print("\nPOST-ENDPOINT OBSERVATIONS")
print("-" * 65)

post_endpoint_df = df[
    df["post_endpoint_observation"]
][
    [
        "mango_id",
        "storage_day",
        "endpoint_day",
        "remaining_shelf_life_days"
    ]
]

if post_endpoint_df.empty:
    print("No post-endpoint observations found.")
else:
    print(post_endpoint_df.to_string(index=False))


# ================================================================
# 13. DISPLAY OUTPUT FILES
# ================================================================

print("\nOUTPUT FILES")
print("-" * 65)

print(f"Target dataset:\n{OUTPUT_FILE}")

print(f"\nEndpoint audit:\n{AUDIT_FILE}")


# ================================================================
# 14. FINAL VALIDATION
# ================================================================

print("\nFINAL VALIDATION")
print("-" * 65)

checks = {
    "All 60 mangoes have SPOILED endpoints":
        mangoes_with_endpoints == total_mangoes,

    "No remaining data-review flags":
        review_flags == 0,

    "No negative shelf-life targets":
        negative_targets == 0,

    "All post-endpoint targets are blank":
        blank_post_endpoint_targets == post_endpoint_count,

    "Output target dataset saved":
        OUTPUT_FILE.exists(),

    "Endpoint audit saved":
        AUDIT_FILE.exists()
}

for check_name, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {check_name}")

if not all(checks.values()):
    raise RuntimeError(
        "One or more final validation checks failed."
    )

print("\n" + "=" * 65)
print("STEP 5 COMPLETED SUCCESSFULLY.")
print("=" * 65)

print("\nOriginal input dataset preserved.")
