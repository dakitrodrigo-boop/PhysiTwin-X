
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ============================================================
# PHYSITWIN-X
# STEP 8: SHAP EXPLAINABILITY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_FILE = RESULTS_DIR / "physitwin_x_xgboost_model.joblib"
TEST_FILE = DATA_DIR / "physitwin_x_test.csv"

GLOBAL_FILE = RESULTS_DIR / "step8_shap_global_importance.csv"
VALUES_FILE = RESULTS_DIR / "step8_shap_values.csv"
INDIVIDUAL_FILE = RESULTS_DIR / "step8_individual_explanations.csv"
REPORT_FILE = RESULTS_DIR / "step8_shap_summary.txt"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GROUP = "mango_id"
TARGET = "remaining_shelf_life_days"

FEATURES = [
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "peel_color",
    "firmness",
    "days_since_first_observation",
    "observation_gap_days",
    "weight_change_g",
    "weight_loss_change_pct",
    "weight_loss_rate_pct_per_day",
    "temperature_change_c",
    "humidity_change_pct"
]

CATEGORICAL_FEATURES = ["peel_color", "firmness"]


# ============================================================
# 1. LOAD MODEL AND TEST DATA
# ============================================================

print("=" * 65)
print("PHYSITWIN-X STEP 8: SHAP EXPLAINABILITY")
print("=" * 65)

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

model_pipeline = joblib.load(MODEL_FILE)
test_df = pd.read_csv(TEST_FILE)

missing = [
    col for col in FEATURES + [GROUP, TARGET]
    if col not in test_df.columns
]

if missing:
    raise ValueError(f"Missing test columns: {missing}")

X_test = test_df[FEATURES].copy()

for col in CATEGORICAL_FEATURES:
    X_test[col] = X_test[col].astype("string")

print("\nMODEL INFORMATION")
print("Model: XGBoost Regression")
print("SHAP version:", shap.__version__)
print("Testing observations:", len(X_test))
print("Testing mangoes:", test_df[GROUP].nunique())


# ============================================================
# 2. TRANSFORM TEST DATA
# ============================================================

print("\nTRANSFORMING TEST FEATURES")

preprocessor = model_pipeline.named_steps["preprocessor"]
xgb_model = model_pipeline.named_steps["regressor"]

X_transformed = preprocessor.transform(X_test)

# SHAP TreeExplainer uses a dense feature matrix here.
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

feature_names = preprocessor.get_feature_names_out()

print("Original predictors:", len(FEATURES))
print("Transformed features:", len(feature_names))


# ============================================================
# 3. CALCULATE SHAP VALUES
# ============================================================

print("\nCALCULATING SHAP VALUES")

explainer = shap.TreeExplainer(xgb_model)

shap_explanation = explainer(X_transformed)

shap_values = np.asarray(shap_explanation.values)

if shap_values.ndim == 3:
    if shap_values.shape[-1] != 1:
        raise ValueError(
            "Unexpected multi-output SHAP values."
        )
    shap_values = shap_values[:, :, 0]

if shap_values.shape != X_transformed.shape:
    raise ValueError(
        "SHAP dimensions do not match transformed features."
    )

print("SHAP calculation completed.")
print("SHAP matrix shape:", shap_values.shape)


# ============================================================
# 4. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCALCULATING GLOBAL FEATURE IMPORTANCE")

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap_value": np.mean(
        np.abs(shap_values), axis=0
    )
})

global_importance = global_importance.sort_values(
    "mean_absolute_shap_value",
    ascending=False
)

global_importance.to_csv(
    GLOBAL_FILE,
    index=False
)


# ============================================================
# 5. SAVE INDIVIDUAL SHAP VALUES
# ============================================================

print("\nSAVING INDIVIDUAL SHAP VALUES")

value_records = []

for row_index in range(len(test_df)):
    for feature_index, feature in enumerate(feature_names):

        value_records.append({
            "mango_id": test_df.iloc[row_index][GROUP],
            "storage_day": test_df.iloc[row_index]["storage_day"],
            "transformed_feature": feature,
            "transformed_feature_value": X_transformed[
                row_index, feature_index
            ],
            "shap_value": shap_values[
                row_index, feature_index
            ]
        })

shap_values_df = pd.DataFrame(value_records)

shap_values_df.to_csv(
    VALUES_FILE,
    index=False
)


# ============================================================
# 6. SAVE INDIVIDUAL MANGO EXPLANATIONS
# ============================================================

print("\nGENERATING INDIVIDUAL EXPLANATIONS")

individual_records = []

for row_index in range(len(test_df)):

    actual = float(test_df.iloc[row_index][TARGET])

    raw_prediction = float(
        model_pipeline.predict(
            X_test.iloc[[row_index]]
        )[0]
    )

    for feature_index, feature in enumerate(feature_names):

        individual_records.append({
            "mango_id": test_df.iloc[row_index][GROUP],
            "storage_day": test_df.iloc[row_index]["storage_day"],
            "actual_remaining_shelf_life_days": actual,
            "raw_predicted_remaining_shelf_life_days": raw_prediction,
            "transformed_feature": feature,
            "shap_value": shap_values[
                row_index, feature_index
            ]
        })

individual_df = pd.DataFrame(individual_records)

individual_df.to_csv(
    INDIVIDUAL_FILE,
    index=False
)


# ============================================================
# 7. GENERATE SHAP SUMMARY PLOT
# ============================================================

print("\nGENERATING SHAP SUMMARY PLOT")

summary_plot = shap.Explanation(
    values=shap_values,
    data=X_transformed,
    feature_names=list(feature_names),
    base_values=np.repeat(
        float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        len(X_test)
    )
)

plt.figure(figsize=(12, 8))

shap.plots.beeswarm(
    summary_plot,
    max_display=20,
    show=False
)

plt.title(
    "PhysiTwin-X SHAP Summary Plot",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "step8_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. GENERATE SHAP BAR PLOT
# ============================================================

print("GENERATING SHAP BAR PLOT")

plt.figure(figsize=(12, 8))

shap.plots.bar(
    summary_plot,
    max_display=20,
    show=False
)

plt.title(
    "PhysiTwin-X SHAP Global Feature Importance",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "step8_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. GENERATE INDIVIDUAL WATERFALL PLOT
# ============================================================

print("GENERATING INDIVIDUAL WATERFALL PLOT")

plt.figure(figsize=(12, 8))

shap.plots.waterfall(
    summary_plot[0],
    max_display=15,
    show=False
)

plt.title(
    "PhysiTwin-X Individual Mango Explanation",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "step8_individual_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. GENERATE SUMMARY REPORT
# ============================================================

report = [
    "=" * 65,
    "PHYSITWIN-X STEP 8 SHAP EXPLAINABILITY REPORT",
    "=" * 65,
    "",
    "MODEL: XGBOOST REGRESSION",
    f"SHAP version: {shap.__version__}",
    f"Testing observations: {len(test_df)}",
    f"Testing mangoes: {test_df[GROUP].nunique()}",
    f"Original predictors: {len(FEATURES)}",
    f"Transformed features: {len(feature_names)}",
    "",
    "TOP 10 GLOBAL SHAP FEATURES"
]

for _, row in global_importance.head(10).iterrows():
    report.append(
        f"{row['feature']}: "
        f"{row['mean_absolute_shap_value']:.6f}"
    )

report.extend([
    "",
    "OUTPUT FILES",
    str(GLOBAL_FILE),
    str(VALUES_FILE),
    str(INDIVIDUAL_FILE),
    str(FIGURES_DIR / "step8_shap_summary.png"),
    str(FIGURES_DIR / "step8_shap_bar.png"),
    str(FIGURES_DIR / "step8_individual_waterfall.png"),
    "",
    "INTERPRETATION NOTE",
    "Positive SHAP values increase the raw model prediction.",
    "Negative SHAP values decrease the raw model prediction.",
    "SHAP values explain model output, not causal effects.",
    "The explanation is in the model's raw prediction scale.",
    "",
    "STEP 8 STATUS: COMPLETED"
])

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TOP 10 SHAP FEATURES")
print("=" * 65)

print(
    global_importance.head(10).to_string(index=False)
)

print("\nOUTPUT FILES")
print("Global importance:", GLOBAL_FILE)
print("SHAP values:", VALUES_FILE)
print("Individual explanations:", INDIVIDUAL_FILE)

print("\nFIGURES")
print("SHAP summary:", FIGURES_DIR / "step8_shap_summary.png")
print("SHAP bar:", FIGURES_DIR / "step8_shap_bar.png")
print(
    "Individual waterfall:",
    FIGURES_DIR / "step8_individual_waterfall.png"
)

print("\nREPORT:", REPORT_FILE)

print("\n" + "=" * 65)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 65)
