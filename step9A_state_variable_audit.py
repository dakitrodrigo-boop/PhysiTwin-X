
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = DATA_DIR / "physitwin_x_endpoint_reverified.csv"

OUTPUT_FILE = RESULTS_DIR / "step9A_state_variable_audit.txt"
CATEGORY_FILE = RESULTS_DIR / "step9A_state_categories.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X STEP 9A: STATE VARIABLE AUDIT")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

# Required observed variables
required = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "peel_color",
    "firmness",
    "ripeness_stage",
    "spoilage_risk"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Sort observations
df = df.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

lines = []

def report(message=""):
    print(message)
    lines.append(str(message))

report("DATASET INFORMATION")
report(f"Total observations: {len(df)}")
report(f"Unique mangoes: {df['mango_id'].nunique()}")
report(f"Storage days: {df['storage_day'].min()} to "
       f"{df['storage_day'].max()}")

report("\nNUMERIC STATE VARIABLES")

numeric_columns = [
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "storage_day"
]

for col in numeric_columns:
    report(f"\n{col}")
    report(str(df[col].describe().round(3)))

report("\nCATEGORICAL STATE VARIABLES")

category_columns = [
    "peel_color",
    "firmness",
    "ripeness_stage",
    "spoilage_risk"
]

category_records = []

for col in category_columns:
    report(f"\n{col}")

    counts = df[col].value_counts(dropna=False)

    for category, count in counts.items():
        report(f"  {category}: {count}")

        category_records.append({
            "variable": col,
            "category": category,
            "frequency": count
        })

# Save category frequencies
pd.DataFrame(category_records).to_csv(
    CATEGORY_FILE,
    index=False
)

# Validate mango-day records
duplicates = df.duplicated(
    subset=["mango_id", "storage_day"]
).sum()

report("\nDATA VALIDATION")
report(f"Duplicate mango-day records: {duplicates}")
report(f"Missing values in required variables: "
       f"{df[required].isna().sum().sum()}")

# Save audit report
OUTPUT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\nSTEP 9A AUDIT COMPLETED")
print(f"Report: {OUTPUT_FILE}")
print(f"Categories: {CATEGORY_FILE}")
