
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_FILE = RESULTS_DIR / "step7_shelf_life_predictions.csv"

OUTPUT_FILE = FIGURES_DIR / "step7_actual_vs_predicted.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
df = pd.read_csv(INPUT_FILE)

actual = df["actual_remaining_shelf_life_days"]
predicted = df["predicted_remaining_shelf_life_days"]

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

# Determine graph limits
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

# Create graph
plt.figure(figsize=(9, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.75,
    s=65,
    label="Test observations"
)

# Perfect prediction reference line
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel(
    "Actual Remaining Shelf Life (days)",
    fontsize=12
)

plt.ylabel(
    "Predicted Remaining Shelf Life (days)",
    fontsize=12
)

plt.title(
    "PhysiTwin-X: Actual vs. Predicted Remaining Shelf Life",
    fontsize=14,
    fontweight="bold"
)

plt.xlim(minimum - 0.5, maximum + 0.5)
plt.ylim(minimum - 0.5, maximum + 0.5)

plt.grid(alpha=0.25)
plt.legend()

metrics_text = (
    f"MAE = {mae:.4f} days\n"
    f"RMSE = {rmse:.4f} days\n"
    f"R² = {r2:.4f}"
)

plt.text(
    0.04,
    0.96,
    metrics_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.15),
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("PHYSITWIN-X ACTUAL VS. PREDICTED GRAPH")
print("=" * 60)

print("Testing observations:", len(df))
print(f"MAE: {mae:.4f} days")
print(f"RMSE: {rmse:.4f} days")
print(f"R²: {r2:.4f}")

print("\nGraph saved successfully:")
print(OUTPUT_FILE)

print("\nSTEP 7A COMPLETED SUCCESSFULLY")
