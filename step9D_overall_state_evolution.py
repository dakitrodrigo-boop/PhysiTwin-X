
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PHYSITWIN-X STEP 9D
# OVERALL STATE EVOLUTION ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_FILE = DATA_DIR / "physitwin_x_dynamic_states.csv"

OUTPUT_CSV = DATA_DIR / "physitwin_x_daily_state_summary.csv"
OUTPUT_REPORT = RESULTS_DIR / "step9D_state_evolution_summary.txt"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHYSITWIN-X STEP 9D: OVERALL STATE EVOLUTION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required = [
    "mango_id",
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "firmness_state_score",
    "ripeness_state_score"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 2. DAILY DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

daily = (
    df.groupby("storage_day")
    .agg(
        mangoes_observed=("mango_id", "nunique"),
        observations=("mango_id", "size"),

        mean_temperature_c=("temperature_c", "mean"),
        sd_temperature_c=("temperature_c", "std"),

        mean_humidity_pct=("relative_humidity_pct", "mean"),
        sd_humidity_pct=("relative_humidity_pct", "std"),

        mean_weight_g=("weight_g", "mean"),
        sd_weight_g=("weight_g", "std"),

        mean_weight_loss_pct=("weight_loss_pct", "mean"),
        sd_weight_loss_pct=("weight_loss_pct", "std"),

        mean_firmness_state=("firmness_state_score", "mean"),
        sd_firmness_state=("firmness_state_score", "std"),

        mean_ripeness_state=("ripeness_state_score", "mean"),
        sd_ripeness_state=("ripeness_state_score", "std"),

        median_weight_loss_pct=("weight_loss_pct", "median"),
        median_firmness_state=("firmness_state_score", "median"),
        median_ripeness_state=("ripeness_state_score", "median")
    )
    .reset_index()
)

daily.to_csv(OUTPUT_CSV, index=False)

# ------------------------------------------------------------
# 3. HELPER FUNCTION FOR DAILY TREND GRAPHS
# ------------------------------------------------------------

def create_trend_graph(
    x,
    mean,
    sd,
    count,
    title,
    ylabel,
    filename
):
    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label="Daily mean"
    )

    # Descriptive mean +/- one standard deviation
    lower = mean - sd
    upper = mean + sd

    plt.fill_between(
        x,
        lower,
        upper,
        alpha=0.15,
        label="Mean ± 1 SD"
    )

    plt.xlabel("Storage day")
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")

    plt.xticks(range(1, 15))
    plt.grid(alpha=0.25)
    plt.legend(loc="best")

    # Show daily number of observed mangoes
    for day, value, n in zip(x, mean, count):
        plt.annotate(
            f"n={int(n)}",
            (day, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7
        )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------------------------------------
# 4. CREATE STATE EVOLUTION GRAPHS
# ------------------------------------------------------------

x = daily["storage_day"]

create_trend_graph(
    x,
    daily["mean_firmness_state"],
    daily["sd_firmness_state"],
    daily["mangoes_observed"],
    "Overall Firmness State Evolution",
    "Mean firmness state score",
    "step9D_firmness_evolution.png"
)

create_trend_graph(
    x,
    daily["mean_ripeness_state"],
    daily["sd_ripeness_state"],
    daily["mangoes_observed"],
    "Overall Ripeness State Evolution",
    "Mean ripeness state score",
    "step9D_ripeness_evolution.png"
)

create_trend_graph(
    x,
    daily["mean_weight_loss_pct"],
    daily["sd_weight_loss_pct"],
    daily["mangoes_observed"],
    "Overall Weight-Loss Evolution",
    "Mean weight loss (%)",
    "step9D_weight_loss_evolution.png"
)

create_trend_graph(
    x,
    daily["mean_weight_g"],
    daily["sd_weight_g"],
    daily["mangoes_observed"],
    "Overall Mango Weight Evolution",
    "Mean weight (g)",
    "step9D_weight_evolution.png"
)

# ------------------------------------------------------------
# 5. ENVIRONMENTAL CONDITION GRAPHS
# ------------------------------------------------------------

create_trend_graph(
    x,
    daily["mean_temperature_c"],
    daily["sd_temperature_c"],
    daily["mangoes_observed"],
    "Overall Storage Temperature Trend",
    "Mean temperature (°C)",
    "step9D_temperature_evolution.png"
)

create_trend_graph(
    x,
    daily["mean_humidity_pct"],
    daily["sd_humidity_pct"],
    daily["mangoes_observed"],
    "Overall Relative Humidity Trend",
    "Mean relative humidity (%)",
    "step9D_humidity_evolution.png"
)

# ------------------------------------------------------------
# 6. VALIDATION
# ------------------------------------------------------------

checks = {
    "Original observation count preserved":
        len(df) == 745,

    "60 unique mangoes":
        df["mango_id"].nunique() == 60,

    "No duplicate mango-day records":
        df.duplicated(["mango_id", "storage_day"]).sum() == 0,

    "14 storage days represented":
        daily["storage_day"].nunique() == 14,

    "No missing daily mean values":
        not daily[
            [
                "mean_firmness_state",
                "mean_ripeness_state",
                "mean_weight_loss_pct",
                "mean_weight_g",
                "mean_temperature_c",
                "mean_humidity_pct"
            ]
        ].isna().any().any()
}

# ------------------------------------------------------------
# 7. SAVE REPORT
# ------------------------------------------------------------

report = [
    "=" * 65,
    "PHYSITWIN-X STEP 9D STATE EVOLUTION SUMMARY",
    "=" * 65,
    f"Total observations: {len(df)}",
    f"Unique mangoes: {df['mango_id'].nunique()}",
    f"Storage days analyzed: {daily['storage_day'].nunique()}",
    "",
    "DAILY OBSERVATION COUNTS",
    daily[
        ["storage_day", "mangoes_observed", "observations"]
    ].to_string(index=False),
    "",
    "DAILY STATE SUMMARY",
    daily[
        [
            "storage_day",
            "mean_firmness_state",
            "mean_ripeness_state",
            "mean_weight_loss_pct",
            "mean_weight_g"
        ]
    ].round(3).to_string(index=False),
    "",
    "VALIDATION"
]

for name, passed in checks.items():
    report.append(f"{name}: {passed}")

report.append(
    f"\nOverall validation passed: {all(checks.values())}"
)

report.extend([
    "",
    "OUTPUT FILES",
    str(OUTPUT_CSV),
    str(OUTPUT_REPORT),
    "",
    "FIGURES",
    "step9D_firmness_evolution.png",
    "step9D_ripeness_evolution.png",
    "step9D_weight_loss_evolution.png",
    "step9D_weight_evolution.png",
    "step9D_temperature_evolution.png",
    "step9D_humidity_evolution.png",
    "",
    "NOTE",
    "Daily averages are descriptive summaries.",
    "Ordinal firmness and ripeness scores are category codes,",
    "not direct physical measurements.",
    "Daily sample counts are displayed to identify changes",
    "in the number of observed mangoes.",
    "",
    "STEP 9D STATUS"
])

OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n".join(report))

if not all(checks.values()):
    raise ValueError("Step 9D validation failed.")

print("\nSTEP 9D COMPLETED SUCCESSFULLY")
