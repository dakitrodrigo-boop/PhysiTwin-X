import pandas as pd
from pathlib import Path

# Load the original Excel workbook
file_path = Path("data/PHYSITWIN-X_MANGODATA.csv.CSV")
df = pd.read_excel(file_path)

print("=" * 50)
print("       PHYSITWIN-X DATASET VALIDATION")
print("=" * 50)

# 1. Dataset dimensions
print("\n1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Missing values
print("\n2. MISSING VALUES")
print(df.isnull().sum())

# 3. Duplicate records
print("\n3. DUPLICATE RECORDS")
print("Exact duplicates:", df.duplicated().sum())

# 4. Mango IDs
print("\n4. MANGO IDENTIFICATION")
print("Unique mangoes:", df["mango_id"].nunique())
print("Mango IDs:", sorted(df["mango_id"].dropna().unique()))

# 5. Storage days
print("\n5. STORAGE DAY RANGE")
print("Minimum day:", df["storage_day"].min())
print("Maximum day:", df["storage_day"].max())

# 6. Spoilage risk categories
print("\n6. SPOILAGE RISK CATEGORIES")
print(df["spoilage_risk"].value_counts(dropna=False))

# 7. Duplicate mango-day combinations
print("\n7. DUPLICATE MANGO-DAY RECORDS")
print(
    df.duplicated(
        subset=["mango_id", "storage_day"]
    ).sum()
)

# 8. Export a proper CSV copy
output_path = Path("data/physitwin_x_mango_dataset.csv")
df.to_csv(output_path, index=False)

print("\n8. CSV EXPORT")
print("Saved to:", output_path)
print("\nDATASET VALIDATION COMPLETE")