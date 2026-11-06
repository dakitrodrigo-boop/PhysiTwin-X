
import pandas as pd
from pathlib import Path

# ==========================================
# PHYSITWIN-X
# VERIFIED CATEGORY DATA CLEANING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "physitwin_x_mango_dataset.csv"

OUTPUT_FILE = BASE_DIR / "data" / "physitwin_x_verified_cleaned.csv"

# Load original CSV
df = pd.read_csv(INPUT_FILE)

# Standardize matching fields without changing original measurements
df["mango_id"] = df["mango_id"].astype(str).str.strip().str.upper()
df["storage_day"] = pd.to_numeric(df["storage_day"], errors="coerce")

# Preserve original category values
if "peel_color_original" not in df.columns:
    df["peel_color_original"] = df["peel_color"]

if "firmness_original" not in df.columns:
    df["firmness_original"] = df["firmness"]

# Create correction log
correction_log = []

def correct_value(mango_id, day, column, expected, corrected):
    global df

    mask = (
        (df["mango_id"] == mango_id) &
        (df["storage_day"] == day)
    )

    matches = df.loc[mask]

    if len(matches) != 1:
        print(
            f"NOT CHANGED: {mango_id}, Day {day} "
            f"- found {len(matches)} matching rows."
        )
        return

    old_value = str(matches.iloc[0][column]).strip()

    # Avoid silently replacing unexpected values
    if old_value != expected:
        print(
            f"REVIEW REQUIRED: {mango_id}, Day {day}, {column}. "
            f"Expected '{expected}', found '{old_value}'."
        )
        return

    df.loc[mask, column] = corrected

    correction_log.append({
        "mango_id": mango_id,
        "storage_day": day,
        "column": column,
        "original_value": old_value,
        "corrected_value": corrected
    })

    print(
        f"CORRECTED: {mango_id}, Day {day}, "
        f"{column}: {old_value} -> {corrected}"
    )


# ==========================================
# A. VERIFIED PEEL-COLOR CORRECTIONS
# ==========================================

correct_value(
    "M10", 9, "peel_color",
    "YELLOW WITH",
    "YELLOW ORANGE WITH FORMING FULL BLACK SPOT"
)

correct_value(
    "M39", 6, "peel_color",
    "YELLOW WITH",
    "YELLOW WITH BROWN SPOT"
)

correct_value(
    "M47", 12, "peel_color",
    "YELLOW-ORANGE WITH",
    "YELLOW ORANGE WITH BLACK SPECKLED"
)

correct_value(
    "M50", 11, "peel_color",
    "YELLOW-ORANG E-",
    "YELLOW-ORANGE-BROWN SPECKLED"
)

correct_value(
    "M55", 5, "peel_color",
    "GREEN-",
    "GREEN WITH YELLOW"
)

correct_value(
    "M58", 6, "peel_color",
    "GREEN-",
    "GREEN WITH YELLOW"
)


# ==========================================
# B. VERIFIED FIRMNESS CORRECTIONS
# ==========================================

correct_value(
    "M10", 13, "firmness",
    "OVERRIPE",
    "MUSHY"
)

correct_value(
    "M39", 10, "firmness",
    "OVERRIPE",
    "VERY SOFT"
)

correct_value(
    "M48", 11, "firmness",
    "SPOILED",
    "MUSHY"
)

# M01 Day 10 is intentionally not changed.
# The original observation supplied was M01 Day 9.
# Verify M01 Day 10 before applying a correction.


# ==========================================
# C. REVIEW FLAGS
# ==========================================

df["firmness_review_required"] = (
    df["firmness"].astype(str).str.upper().isin(
        ["SPOILED", "OVERRIPE"]
    )
)

df["peel_color_review_required"] = (
    df["peel_color"].astype(str).str.upper().str.endswith(
        (" WITH", "GREEN-", "YELLOW-")
    )
)


# ==========================================
# D. SAVE CLEANED DATASET
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)

log_df = pd.DataFrame(correction_log)

log_file = BASE_DIR / "results" / "verified_correction_log.csv"
log_file.parent.mkdir(parents=True, exist_ok=True)
log_df.to_csv(log_file, index=False)


# ==========================================
# E. FINAL REPORT
# ==========================================

print("\n==========================================")
print("PHYSITWIN-X VERIFIED CLEANING REPORT")
print("==========================================")

print("Original dataset:", INPUT_FILE.name)
print("Cleaned dataset:", OUTPUT_FILE.name)

print("\nTotal verified corrections:", len(correction_log))

print("\nCorrection summary:")
if not log_df.empty:
    print(log_df.to_string(index=False))
else:
    print("No corrections were applied.")

print("\nRemaining firmness entries requiring review:")
print(
    df.loc[
        df["firmness_review_required"],
        ["mango_id", "storage_day", "firmness", "ripeness_stage"]
    ].to_string(index=False)
)

print("\nRemaining peel-color entries requiring review:")
print(
    df.loc[
        df["peel_color_review_required"],
        ["mango_id", "storage_day", "peel_color"]
    ].to_string(index=False)
)

print("\nCLEANING COMPLETED.")
print("Original CSV was not overwritten.")
