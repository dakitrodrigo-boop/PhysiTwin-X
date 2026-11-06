import pandas as pd

df = pd.read_csv("data/physitwin_x_mango_dataset.csv")

print("=" * 55)
print("PHYSITWIN-X CATEGORY INSPECTION")
print("=" * 55)

# All peel-color labels and frequencies
print("\n1. PEEL COLOR LABELS")

print(
    df["peel_color"]
    .value_counts(dropna=False)
    .to_string()
)

# Firmness categories and frequencies
print("\n2. FIRMNESS CATEGORIES")

print(
    df["firmness"]
    .value_counts(dropna=False)
    .to_string()
)

# Inspect records with questionable firmness values
print("\n3. RECORDS WITH SPOILED OR OVERRIPE FIRMNESS")

questionable = df[
    df["firmness"].astype(str).str.upper().isin(
        ["SPOILED", "OVERRIPE"]
    )
]

print(
    questionable[
        [
            "mango_id",
            "storage_day",
            "peel_color",
            "firmness",
            "ripeness_stage",
            "spoilage_risk"
        ]
    ].to_string(index=False)
)

print("\nCATEGORY INSPECTION COMPLETE")