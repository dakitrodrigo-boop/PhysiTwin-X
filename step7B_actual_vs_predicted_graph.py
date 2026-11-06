
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# PHYSITWIN-X
# STEP 7B: ACTUAL VS. PREDICTED VISUALIZATION
# =====================================================

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("results/step7_shelf_life_predictions.csv")

actual = df["actual_remaining_shelf_life_days"].to_numpy()
raw = df["predicted_remaining_shelf_life_days"].to_numpy()
clipped = np.maximum(raw, 0)

# Calculate metrics
def get_metrics(y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    return mae, rmse, r2

raw_metrics = get_metrics(actual, raw)
clipped_metrics = get_metrics(actual, clipped)

# Common axis range
lower = min(0, raw.min()) - 0.3
upper = max(actual.max(), raw.max(), clipped.max()) + 1

# -----------------------------------------------------
# FIGURE 1: RAW XGBOOST PREDICTIONS
# -----------------------------------------------------

plt.figure(figsize=(9, 7))

plt.scatter(
    actual,
    raw,
    alpha=0.75,
    s=55,
    label="XGBoost Predictions"
)

plt.plot(
    [0, upper],
    [0, upper],
    linestyle="--",
    label="Perfect Prediction (y = x)"
)

plt.xlabel("Actual Remaining Shelf Life (days)")
plt.ylabel("Predicted Remaining Shelf Life (days)")
plt.title("Actual vs. Predicted Remaining Shelf Life Using XGBoost")

plt.xlim(lower, upper)
plt.ylim(lower, upper)
plt.grid(alpha=0.25)
plt.legend()

plt.text(
    0.04, 0.96,
    f"MAE = {raw_metrics[0]:.3f} days\n"
    f"RMSE = {raw_metrics[1]:.3f} days\n"
    f"R² = {raw_metrics[2]:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    "results/figures/figure_11_actual_vs_predicted_raw.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------
# FIGURE 2: NONNEGATIVE OPERATIONAL PREDICTIONS
# -----------------------------------------------------

plt.figure(figsize=(9, 7))

plt.scatter(
    actual,
    clipped,
    alpha=0.75,
    s=55,
    label="Nonnegative XGBoost Predictions"
)

plt.plot(
    [0, upper],
    [0, upper],
    linestyle="--",
    label="Perfect Prediction (y = x)"
)

plt.xlabel("Actual Remaining Shelf Life (days)")
plt.ylabel("Nonnegative Predicted Remaining Shelf Life (days)")
plt.title("Actual vs. Predicted Remaining Shelf Life (Operational Output)")

plt.xlim(0, upper)
plt.ylim(0, upper)
plt.grid(alpha=0.25)
plt.legend()

plt.text(
    0.04, 0.96,
    f"MAE = {clipped_metrics[0]:.3f} days\n"
    f"RMSE = {clipped_metrics[1]:.3f} days\n"
    f"R² = {clipped_metrics[2]:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    "results/figures/figure_11B_actual_vs_predicted_operational.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("PHYSITWIN-X: STEP 7B COMPLETED")
print("=" * 60)

print("\nRAW XGBOOST GRAPH")
print("MAE:", round(raw_metrics[0], 4))
print("RMSE:", round(raw_metrics[1], 4))
print("R2:", round(raw_metrics[2], 4))

print("\nNONNEGATIVE OPERATIONAL GRAPH")
print("MAE:", round(clipped_metrics[0], 4))
print("RMSE:", round(clipped_metrics[1], 4))
print("R2:", round(clipped_metrics[2], 4))

print("\nGENERATED FIGURES:")
print("figure_11_actual_vs_predicted_raw.png")
print("figure_11B_actual_vs_predicted_operational.png")

print("\nLocation: results/figures/")
