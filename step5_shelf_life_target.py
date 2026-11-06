
from pathlib import Path
import pandas as pd

# ============================================================
# PHYSITWIN-X STEP 5E
# SHELF-LIFE TARGET GENERATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "physitwin_x_endpoint_reverified.csv"
OUTPUT_FILE = BASE_DIR / "data" / "physitwin_x_shelf_life_targets.csv"
SUMMARY_FILE = BASE_DIR / "results" / "step5_shelf_life_target_summary.txt"

print("=" * 65)
print("PHYSITWIN-X STEP 5E: SHELF-LIFE TARGET GENERATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD VERIFIED DATASET
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "mango_id",
    "storage_day",
    "ripeness_stage"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if len(df) != 745:
    raise ValueError(f"Expected 745 observations, found {len(df)}")

if df["mango_id"].nunique() != 60:
    raise ValueError("Expected 60 mangoes.")

df["storage_day"] = pd.to_numeric(
    df["storage_day"], errors="raise"
)

print("\nINPUT DATASET")
print(f"Observations: {len(df)}")
print(f"Mangoes: {df['mango_id'].nunique()}")

# ------------------------------------------------------------
# 2. IDENTIFY FIRST SPOILED DAY PER MANGO
# ------------------------------------------------------------

df["ripeness_stage"] = (
    df["ripeness_stage"].astype(str).str.strip().str.upper()
)

spoiled = df[df["ripeness_stage"] == "SPOILED"].copy()

if spoiled.empty:
    raise ValueError("No SPOILED observations found.")

endpoint_days = (
    spoiled.groupby("mango_id")["storage_day"]
    .min()
    .rename("endpoint_day")
)

if endpoint_days.nunique() == 0:
    raise ValueError("Endpoint days could not be calculated.")

if len(endpoint_days) != 60:
    raise ValueError(
        f"Expected 60 mango endpoints, found {len(endpoint_days)}."
    )

# ------------------------------------------------------------
# 3. ADD ENDPOINT DAY
# ------------------------------------------------------------

df = df.merge(
    endpoint_days,
    on="mango_id",
    how="left",
    validate="many_to_one"
)

if df["endpoint_day"].isna().any():
    raise ValueError("Some mangoes have no verified endpoint.")

# ------------------------------------------------------------
# 4. CALCULATE REMAINING SHELF LIFE
# ------------------------------------------------------------

df["remaining_shelf_life_days"] = (
    df["endpoint_day"] - df["storage_day"]
)

if (df["remaining_shelf_life_days"] < 0).any():
    raise ValueError("Negative shelf-life targets detected.")

if df["remaining_shelf_life_days"].isna().any():
    raise ValueError("Missing shelf-life targets detected.")

# ------------------------------------------------------------
# 5. VALIDATE ENDPOINTS
# ------------------------------------------------------------

endpoint_check = df[
    df["storage_day"] == df["endpoint_day"]
]

if len(endpoint_check) != 60:
    raise ValueError(
        "Expected one endpoint observation per mango."
    )

if not endpoint_check["ripeness_stage"].eq("SPOILED").all():
    raise ValueError(
        "Some endpoint observations are not classified as SPOILED."
    )

print("\nSHELF-LIFE TARGET SUMMARY")
print(f"Total observations: {len(df)}")
print(f"Total mangoes: {df['mango_id'].nunique()}")
print(f"Endpoint observations: {len(endpoint_check)}")
print(
    "Minimum remaining shelf life:",
    df["remaining_shelf_life_days"].min()
)
print(
    "Maximum remaining shelf life:",
    df["remaining_shelf_life_days"].max()
)

# ------------------------------------------------------------
# 6. SAVE DATASET
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

summary = [
    "PHYSITWIN-X STEP 5E: SHELF-LIFE TARGET SUMMARY",
    "=" * 55,
    f"Total observations: {len(df)}",
    f"Total mangoes: {df['mango_id'].nunique()}",
    f"Endpoint observations: {len(endpoint_check)}",
    f"Minimum remaining shelf life: {df['remaining_shelf_life_days'].min()}",
    f"Maximum remaining shelf life: {df['remaining_shelf_life_days'].max()}",
    f"Mean remaining shelf life: {df['remaining_shelf_life_days'].mean():.3f}",
    "",
    "Formula:",
    "Remaining shelf life = First spoiled day - Current storage day",
    "",
    f"Input: {INPUT_FILE}",
    f"Output: {OUTPUT_FILE}",
    "",
    "Original input dataset preserved."
]

SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8"
)

print("\nOUTPUT FILES")
print(f"Target dataset: {OUTPUT_FILE}")
print(f"Summary report: {SUMMARY_FILE}")

print("\nOriginal input dataset preserved.")
print("=" * 65)
print("STEP 5E COMPLETED SUCCESSFULLY.")
print("=" * 65)
