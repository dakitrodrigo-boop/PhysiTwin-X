
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PHYSITWIN-X STEP 9C
# INDIVIDUAL MANGO STATE TRAJECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_FILE = DATA_DIR / "physitwin_x_dynamic_states.csv"

INDIVIDUAL_DIR = FIGURES_DIR / "individual_mango_trajectories"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X STEP 9C: INDIVIDUAL MANGO TRAJECTORIES")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "mango_id",
    "storage_day",
    "weight_loss_pct",
    "firmness_state_score",
    "ripeness_state_score"
]

missing = [
    col for col in required_columns if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

mango_ids = sorted(df["mango_id"].unique())

print(f"Observations: {len(df)}")
print(f"Individual mangoes: {len(mango_ids)}")

# ------------------------------------------------------------
# 2. DEFINE STATE LABELS
# ------------------------------------------------------------

firmness_labels = [
    "Very firm",
    "Firm",
    "Moderately firm",
    "Medium firm",
    "Slightly firm",
    "Soft",
    "Very soft",
    "Mushy"
]

ripeness_labels = [
    "Unripe",
    "Mature green",
    "Breaker",
    "Ripe",
    "Fully ripe",
    "Overripe",
    "Spoiled"
]

# ------------------------------------------------------------
# 3. CREATE OVERVIEW: ALL MANGOES
# ------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(13, 16))

for mango_id, group in df.groupby("mango_id"):

    group = group.sort_values("storage_day")

    axes[0].plot(
        group["storage_day"],
        group["firmness_state_score"],
        marker=".",
        markersize=3,
        linewidth=1,
        alpha=0.45
    )

    axes[1].plot(
        group["storage_day"],
        group["ripeness_state_score"],
        marker=".",
        markersize=3,
        linewidth=1,
        alpha=0.45
    )

    axes[2].plot(
        group["storage_day"],
        group["weight_loss_pct"],
        marker=".",
        markersize=3,
        linewidth=1,
        alpha=0.45
    )

axes[0].set_title(
    "Firmness State Trajectories of 60 Mangoes"
)
axes[0].set_ylabel("Firmness state score")
axes[0].set_yticks(range(8))
axes[0].set_yticklabels(firmness_labels)

axes[1].set_title(
    "Ripeness State Trajectories of 60 Mangoes"
)
axes[1].set_ylabel("Ripeness state score")
axes[1].set_yticks(range(7))
axes[1].set_yticklabels(ripeness_labels)

axes[2].set_title(
    "Weight-Loss Trajectories of 60 Mangoes"
)
axes[2].set_ylabel("Weight loss (%)")
axes[2].set_xlabel("Storage day")

for ax in axes:
    ax.set_xticks(range(1, 15))
    ax.grid(alpha=0.25)

fig.suptitle(
    "PHYSITWIN-X: Individual Mango State Evolution",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

overview_file = FIGURES_DIR / "step9C_all_mango_trajectories.png"

plt.savefig(
    overview_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 4. CREATE INDIVIDUAL GRAPHS FOR EACH MANGO
# ------------------------------------------------------------

for mango_id, group in df.groupby("mango_id"):

    group = group.sort_values("storage_day")

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))

    axes[0].plot(
        group["storage_day"],
        group["firmness_state_score"],
        marker="o",
        linewidth=2
    )

    axes[0].set_title("Firmness State")
    axes[0].set_ylabel("State score")
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels(firmness_labels)

    axes[1].plot(
        group["storage_day"],
        group["ripeness_state_score"],
        marker="o",
        linewidth=2
    )

    axes[1].set_title("Ripeness State")
    axes[1].set_ylabel("State score")
    axes[1].set_yticks(range(7))
    axes[1].set_yticklabels(ripeness_labels)

    axes[2].plot(
        group["storage_day"],
        group["weight_loss_pct"],
        marker="o",
        linewidth=2
    )

    axes[2].set_title("Weight Loss")
    axes[2].set_ylabel("Weight loss (%)")
    axes[2].set_xlabel("Storage day")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(range(1, 15))

    fig.suptitle(
        f"PhysiTwin-X Individual State Trajectory: {mango_id}",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()

    output_file = INDIVIDUAL_DIR / f"{mango_id}_trajectory.png"

    plt.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# 5. SAVE INDIVIDUAL TRAJECTORY DATA
# ------------------------------------------------------------

trajectory_columns = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "firmness",
    "firmness_state_score",
    "ripeness_stage",
    "ripeness_state_score",
    "temperature_change_c",
    "humidity_change_pct",
    "weight_change_g",
    "weight_loss_change_pct",
    "firmness_state_change",
    "ripeness_state_change"
]

available_columns = [
    col for col in trajectory_columns if col in df.columns
]

trajectory_data = df[available_columns].copy()

trajectory_file = DATA_DIR / "physitwin_x_individual_trajectories.csv"

trajectory_data.to_csv(
    trajectory_file,
    index=False
)

# ------------------------------------------------------------
# 6. VALIDATION AND SUMMARY
# ------------------------------------------------------------

individual_graphs = list(INDIVIDUAL_DIR.glob("*_trajectory.png"))

summary = [
    "=" * 65,
    "PHYSITWIN-X STEP 9C SUMMARY",
    "=" * 65,
    f"Total observations: {len(df)}",
    f"Unique mangoes: {len(mango_ids)}",
    f"Overview graph created: {overview_file.exists()}",
    f"Individual graphs created: {len(individual_graphs)}",
    f"Trajectory data rows: {len(trajectory_data)}",
    f"Duplicate mango-day records: "
    f"{df.duplicated(['mango_id', 'storage_day']).sum()}",
    "",
    "OUTPUT FILES",
    str(overview_file),
    str(trajectory_file),
    str(INDIVIDUAL_DIR),
    "",
    "STEP 9C STATUS: COMPLETED"
]

summary_file = RESULTS_DIR / "step9C_trajectory_summary.txt"

summary_file.write_text(
    "\n".join(summary),
    encoding="utf-8"
)

print("\n".join(summary))
print("\nSTEP 9C COMPLETED SUCCESSFULLY")
