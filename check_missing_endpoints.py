
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

files = [
    "physitwin_x_mango_dataset.csv",
    "physitwin_x_verified_cleaned.csv",
    "physitwin_x_ripeness_verified.csv"
]

missing_mangoes = ["M01", "M21", "M25", "M40"]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("=" * 60)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path)

    selected = df[df["mango_id"].isin(missing_mangoes)]

    columns = [
        col for col in [
            "mango_id",
            "storage_day",
            "ripeness_stage",
            "peel_color",
            "firmness",
            "spoilage_risk",
            "data_review_required"
        ]
        if col in selected.columns
    ]

    for mango in missing_mangoes:
        print(f"\n--- {mango} ---")

        records = selected[selected["mango_id"] == mango]

        print(
            records[columns].to_string(index=False)
        )

print("\nInspection completed.")
