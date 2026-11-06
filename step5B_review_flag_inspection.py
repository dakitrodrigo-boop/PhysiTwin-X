
from pathlib import Path
import pandas as pd

# =====================================================
# PHYSITWIN-X STEP 5B
# REVIEW FLAG INSPECTION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "physitwin_x_endpoint_verified.csv"
OUTPUT_FILE = BASE_DIR / "results" / "step5B_review_flag_inspection.csv"
SUMMARY_FILE = BASE_DIR / "results" / "step5B_review_flag_summary.txt"

# Create results folder if needed
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHYSITWIN-X STEP 5B: REVIEW FLAG INSPECTION")
print("=" * 60)

# -----------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nINPUT DATASET")
print(f"File: {INPUT_FILE}")
print(f"Observations: {len(df)}")
print(f"Mangoes: {df['mango_id'].nunique()}")

# -----------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# -----------------------------------------------------

required_columns = [
    "mango_id",
    "storage_day",
    "firmness",
    "ripeness_stage",
    "spoilage_risk"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# -----------------------------------------------------
# 3. IDENTIFY REVIEW FLAG COLUMN
# -----------------------------------------------------

review_candidates = [
    "review_flag",
    "needs_review",
    "manual_review_flag"
]

review_column = next(
    (col for col in review_candidates if col in df.columns),
    None
)

if review_column is None:
    df["review_flag"] = False
    review_column = "review_flag"
    print("\nNo existing review flag found.")
    print("A new review flag column was created.")

# -----------------------------------------------------
# 4. STANDARDIZE REVIEW FLAGS
# -----------------------------------------------------

def is_review_required(value):

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().upper() in [
        "TRUE",
        "YES",
        "Y",
        "1",
        "REVIEW"
    ]


df["review_flag"] = df[review_column].apply(
    is_review_required
)

# -----------------------------------------------------
# 5. INSPECT RECORDS REQUIRING REVIEW
# -----------------------------------------------------

review_records = df[df["review_flag"]].copy()

print("\nREVIEW FLAG SUMMARY")
print(f"Records requiring review: {len(review_records)}")
print(f"Records not flagged: {len(df) - len(review_records)}")

if len(review_records) > 0:

    print("\nRECORDS REQUIRING REVIEW")

    print(
        review_records[
            [
                "mango_id",
                "storage_day",
                "firmness",
                "ripeness_stage",
                "spoilage_risk"
            ]
        ].to_string(index=False)
    )

else:
    print("\nNo records are currently flagged.")

# -----------------------------------------------------
# 6. CHECK VERIFIED ENDPOINTS
# -----------------------------------------------------

endpoint_records = df[
    df["ripeness_stage"]
    .astype(str)
    .str.strip()
    .str.upper()
    .eq("SPOILED")
].copy()

print("\nVERIFIED SPOILED RECORDS")
print(f"Total spoiled observations: {len(endpoint_records)}")

# -----------------------------------------------------
# 7. SAVE INSPECTION REPORT
# -----------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

summary = [
    "PHYSITWIN-X STEP 5B REVIEW FLAG INSPECTION",
    "=" * 55,
    f"Total observations: {len(df)}",
    f"Total mangoes: {df['mango_id'].nunique()}",
    f"Review records: {len(review_records)}",
    f"Non-review records: {len(df) - len(review_records)}",
    f"Verified spoiled observations: {len(endpoint_records)}",
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
print(f"Inspection dataset: {OUTPUT_FILE}")
print(f"Summary report: {SUMMARY_FILE}")

print("\nOriginal input dataset preserved.")
print("STEP 5B COMPLETED SUCCESSFULLY.")
