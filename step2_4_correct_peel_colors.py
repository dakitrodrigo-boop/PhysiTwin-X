
import os
import pandas as pd

# ============================================================
# PHYSITWIN-X
# STEP 2.4: VERIFIED PEEL-COLOR CORRECTIONS
# ============================================================

INPUT_FILE = "data/physitwin_x_verified_cleaned.csv"
OUTPUT_FILE = "data/physitwin_x_verified_cleaned_v2.csv"
LOG_FILE = "results/step2_4_peel_color_correction_log.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X")
print("STEP 2.4: VERIFIED PEEL-COLOR CORRECTIONS")
print("=" * 65)

# Load the previously verified dataset
df = pd.read_csv(INPUT_FILE)

required_columns = ["mango_id", "storage_day", "peel_color"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Verified corrections based on the original observation sheets
corrections = [
    {
        "mango_id": "M55",
        "storage_day": 5,
        "new_peel_color": "GREEN-YELLOW"
    },
    {
        "mango_id": "M58",
        "storage_day": 6,
        "new_peel_color": "GREEN-YELLOW"
    }
]

correction_log = []

for item in corrections:
    mango_id = item["mango_id"]
    day = item["storage_day"]
    new_value = item["new_peel_color"]

    mask = (
        (df["mango_id"].astype(str).str.strip() == mango_id)
        & (pd.to_numeric(df["storage_day"], errors="coerce") == day)
    )

    matches = df.index[mask].tolist()

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one record for {mango_id}, Day {day}; "
            f"found {len(matches)}. No output dataset was saved."
        )

    row_index = matches[0]
    old_value = df.at[row_index, "peel_color"]

    correction_log.append({
        "mango_id": mango_id,
        "storage_day": day,
        "previous_peel_color": old_value,
        "corrected_peel_color": new_value,
        "correction_basis": "Verified against original observation sheet"
    })

    df.at[row_index, "peel_color"] = new_value

# Save corrected dataset separately
df.to_csv(OUTPUT_FILE, index=False)

# Save an audit log
log_df = pd.DataFrame(correction_log)
log_df.to_csv(LOG_FILE, index=False)

# Verify the saved corrections
print("\nVERIFIED CORRECTIONS")

for item in corrections:
    check = df[
        (df["mango_id"].astype(str).str.strip() == item["mango_id"])
        & (pd.to_numeric(df["storage_day"], errors="coerce")
           == item["storage_day"])
    ]

    print(
        item["mango_id"],
        "| Day", item["storage_day"],
        "| Peel color:", check.iloc[0]["peel_color"]
    )

print("\nDATASET INFORMATION")
print("Total observations:", len(df))
print("Total mangoes:", df["mango_id"].nunique())

print("\nSAVED FILES")
print(OUTPUT_FILE)
print(LOG_FILE)

print("\nOriginal input dataset was not overwritten.")
print("STEP 2.4 COMPLETED SUCCESSFULLY.")
