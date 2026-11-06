
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PHYSITWIN-X
# STEP 7A: PREDICTION DIAGNOSTICS
# ============================================================

os.makedirs("results", exist_ok=True)

input_file = "results/step7_shelf_life_predictions.csv"

df = pd.read_csv(input_file)

actual = df["actual_remaining_shelf_life_days"]
raw = df["predicted_remaining_shelf_life_days"]

# Apply nonnegative operational constraint
clipped = np.maximum(raw, 0)

# Evaluation function
def calculate_metrics(y_true, y_pred, model_name):
    return {
        "model": model_name,
        "MAE_days": mean_absolute_error(y_true, y_pred),
        "RMSE_days": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

results = [
    calculate_metrics(actual, raw, "Raw XGBoost"),
    calculate_metrics(actual, clipped, "Nonnegative Operational XGBoost")
]

metrics = pd.DataFrame(results)

# Negative prediction information
negative = df[raw < 0].copy()

negative_count = len(negative)
total_count = len(df)
negative_percentage = (negative_count / total_count) * 100

# Per-mango performance
df["clipped_prediction"] = clipped
df["clipped_absolute_error"] = np.abs(actual - clipped)

per_mango = df.groupby("mango_id").agg(
    observations=("actual_remaining_shelf_life_days", "count"),
    MAE_raw=("absolute_error_days", "mean"),
    MAE_clipped=("clipped_absolute_error", "mean"),
    negative_predictions=(
        "predicted_remaining_shelf_life_days",
        lambda x: int((x < 0).sum())
    )
).reset_index()

# Save diagnostics
metrics.to_csv(
    "results/step7A_prediction_metrics.csv",
    index=False
)

negative.to_csv(
    "results/step7A_negative_predictions.csv",
    index=False
)

per_mango.to_csv(
    "results/step7A_per_mango_performance.csv",
    index=False
)

# Print report
print("=" * 65)
print("PHYSITWIN-X")
print("STEP 7A: PREDICTION DIAGNOSTICS")
print("=" * 65)

print("\nPREDICTION COUNTS")
print("Total test observations:", total_count)
print("Negative raw predictions:", negative_count)
print(f"Negative prediction percentage: {negative_percentage:.2f}%")

print("\nMODEL PERFORMANCE COMPARISON")
print(metrics.to_string(index=False))

print("\nNEGATIVE PREDICTION DETAILS")
if negative_count:
    print(
        negative[
            [
                "mango_id",
                "storage_day",
                "actual_remaining_shelf_life_days",
                "predicted_remaining_shelf_life_days"
            ]
        ].to_string(index=False)
    )
else:
    print("No negative predictions.")

print("\nPER-MANGO PERFORMANCE")
print(per_mango.to_string(index=False))

print("\nSAVED FILES")
print("results/step7A_prediction_metrics.csv")
print("results/step7A_negative_predictions.csv")
print("results/step7A_per_mango_performance.csv")

print("\nSTEP 7A COMPLETED SUCCESSFULLY.")
