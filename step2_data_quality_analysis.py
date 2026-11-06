import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load validated CSV
df = pd.read_csv("data/physitwin_x_mango_dataset.csv")

print("=" * 50)
print("PHYSITWIN-X DATA QUALITY ANALYSIS")
print("=" * 50)

# 1. Observations per mango
print("\n1. OBSERVATIONS PER MANGO")
print(df.groupby("mango_id").size().describe())

print("\nMangoes with unexpected observation counts:")
counts = df.groupby("mango_id").size()
print(counts[counts != 14])

# 2. Storage days per mango
print("\n2. STORAGE DAY COVERAGE")
day_counts = df.groupby("mango_id")["storage_day"].agg(
    ["min", "max", "count", "nunique"]
)
print(day_counts)

# 3. Numeric variable summary
print("\n3. NUMERIC VARIABLE SUMMARY")
print(df.describe())

# 4. Category distribution
print("\n4. RIPENESS STAGE")
print(df["ripeness_stage"].value_counts(dropna=False))

print("\n5. PEEL COLOR")
print(df["peel_color"].value_counts(dropna=False))

print("\n6. FIRMNESS")
print(df["firmness"].value_counts(dropna=False))

# 5. Save results
Path("results").mkdir(exist_ok=True)

df.describe().to_csv(
    "results/numeric_summary.csv"
)

day_counts.to_csv(
    "results/mango_observation_counts.csv"
)

print("\nDATA QUALITY ANALYSIS COMPLETE")