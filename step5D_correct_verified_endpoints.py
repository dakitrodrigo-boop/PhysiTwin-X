
from pathlib import Path
import pandas as pd

# ============================================================
# PHYSITWIN-X STEP 5D
# VERIFIED ENDPOINT CORRECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "physitwin_x_review_verified.csv"
OUTPUT_FILE = BASE_DIR / "data" / "physitwin_x_endpoint_reverified.csv"
LOG_FILE = BASE_DIR / "results" / "step5D_endpoint_correction_log.csv"

print("=" * 65)
print("PHYSITWIN-X STEP 5D: VERIFIED ENDPOINT CORRECTION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "mango_id",
    "storage_day",
    "firmness",
    "ripeness_stage",
    "spoilage_risk"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if len(df) != 745:
    raise ValueError(f"Expected 745 observations, found {len(df)}")

if df["mango_id"].nunique() != 60:
    raise ValueError("Expected exactly 60 unique mangoes.")

print("\nINPUT DATASET")
print(f"Observations: {len(df)}")
print(f"Mangoes: {df['mango_id'].nunique()}")

# ------------------------------------------------------------
# 2. VERIFIED CORRECTIONS
# Based on the original observation records
# ------------------------------------------------------------

corrections = [

    # M10 VERIFIED ENDPOINTS
    {
        "mango_id": "M10",
        "storage_day": 13,
        "firmness": "VERY SOFT",
        "ripeness_stage": "OVERRIPE",
        "spoilage_risk": "HIGH RISK"
    },
    {
        "mango_id": "M10",
        "storage_day": 14,
        "firmness": "MUSHY",
        "ripeness_stage": "SPOILED",
        "spoilage_risk": "VERY HIGH RISK"
    },

    # M43 VERIFIED ENDPOINTS
    {
        "mango_id": "M43",
        "storage_day": 10,
        "firmness": "VERY SOFT",
        "ripeness_stage": "OVERRIPE",
        "spoilage_risk": "HIGH RISK"
    },
    {
        "mango_id": "M43",
        "storage_day": 11,
        "firmness": "VERY SOFT",
        "ripeness_stage": "OVERRIPE",
        "spoilage_risk": "HIGH RISK"
    },
    {
        "mango_id": "M43",
        "storage_day": 12,
        "firmness": "MUSHY",
        "ripeness_stage": "SPOILED",
        "spoilage_risk": "VERY HIGH RISK"
    },

    # M60 VERIFIED ENDPOINTS
    {
        "mango_id": "M60",
        "storage_day": 7,
        "firmness": "VERY SOFT",
        "ripeness_stage": "OVERRIPE",
        "spoilage_risk": "HIGH RISK"
    },
    {
        "mango_id": "M60",
        "storage_day": 8,
        "firmness": "MUSHY",
        "ripeness_stage": "SPOILED",
        "spoilage_risk": "VERY HIGH RISK"
    }
]

# ------------------------------------------------------------
# 3. APPLY VERIFIED CORRECTIONS
# ------------------------------------------------------------

correction_logs = []

for item in corrections:

    mask = (
        df["mango_id"].astype(str).str.strip().eq(item["mango_id"])
        & pd.to_numeric(
            df["storage_day"], errors="coerce"
        ).eq(item["storage_day"])
    )

    matches = df.index[mask].tolist()

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one record for "
            f"{item['mango_id']} Day {item['storage_day']}. "
            f"Found: {len(matches)}"
        )

    idx = matches[0]

    old_values = {
        column: df.at[idx, column]
        for column in [
            "firmness",
            "ripeness_stage",
            "spoilage_risk"
        ]
    }

    # Apply verified values
    for column in [
        "firmness",
        "ripeness_stage",
        "spoilage_risk"
    ]:
        df.at[idx, column] = item[column]

    correction_logs.append({
        "mango_id": item["mango_id"],
        "storage_day": item["storage_day"],
        "old_firmness": old_values["firmness"],
        "new_firmness": item["firmness"],
        "old_ripeness": old_values["ripeness_stage"],
        "new_ripeness": item["ripeness_stage"],
        "old_spoilage_risk": old_values["spoilage_risk"],
        "new_spoilage_risk": item["spoilage_risk"]
    })

    print(
        f"VERIFIED: {item['mango_id']} Day {item['storage_day']} | "
        f"{item['firmness']} | "
        f"{item['ripeness_stage']} | "
        f"{item['spoilage_risk']}"
    )

# ------------------------------------------------------------
# 4. DATASET VALIDATION
# ------------------------------------------------------------

spoiled = df[
    df["ripeness_stage"]
    .astype(str)
    .str.strip()
    .str.upper()
    .eq("SPOILED")
]

spoiled_count = len(spoiled)
unique_spoiled_mangoes = spoiled["mango_id"].nunique()

print("\nFINAL VALIDATION")
print("-" * 45)
print(f"Total observations: {len(df)}")
print(f"Total mangoes: {df['mango_id'].nunique()}")
print(f"Spoiled observations: {spoiled_count}")
print(f"Unique spoiled mangoes: {unique_spoiled_mangoes}")

duplicates = df.duplicated(
    subset=["mango_id", "storage_day"]
)

if duplicates.any():
    raise ValueError("Duplicate mango/day records detected.")

# Verify first spoiled day for each mango
first_spoiled = (
    spoiled.groupby("mango_id")["storage_day"].min()
)

print("\nFIRST SPOILED DAY PER MANGO")
print(first_spoiled.to_string())

# Stop if the count does not match.
# Do not save an unverified dataset.
if spoiled_count != 60 or unique_spoiled_mangoes != 60:
    raise ValueError(
        "Validation failed. Expected 60 spoiled observations "
        "across 60 mangoes. Output was NOT saved."
    )

# ------------------------------------------------------------
# 5. SAVE VERIFIED DATASET AND CORRECTION LOG
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

pd.DataFrame(correction_logs).to_csv(
    LOG_FILE,
    index=False
)

print("\nOUTPUT FILES")
print(f"Verified dataset: {OUTPUT_FILE}")
print(f"Correction log: {LOG_FILE}")

print("\nOriginal input dataset preserved.")
print("=" * 65)
print("STEP 5D COMPLETED SUCCESSFULLY.")
print("=" * 65)
