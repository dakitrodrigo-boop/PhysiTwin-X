
# ============================================================
# PHYSITWIN-X STEP 12
# DIGITAL TWIN INTEGRATION
#
# Integrates:
# 1. Step 7 XGBoost shelf-life regression
# 2. Step 10C XGBoost spoilage-risk classification
# 3. Step 9 dynamic mango states
# 4. Step 11 SHAP explainability reference
#
# IMPORTANT:
# - Does not retrain either model.
# - Does not overwrite original datasets or models.
# - Creates separate integration outputs.
# - Existing observations are used for an integration run.
# ============================================================

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INTEGRATION_DIR = RESULTS_DIR / "step12_digital_twin"

INTEGRATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Existing trained models
SHELF_LIFE_MODEL_FILE = (
    RESULTS_DIR / "physitwin_x_xgboost_model.joblib"
)

RISK_MODEL_FILE = (
    RESULTS_DIR / "physitwin_x_spoilage_risk_model.joblib"
)

# Dynamic-state dataset
DYNAMIC_STATE_FILE = (
    DATA_DIR / "physitwin_x_dynamic_states.csv"
)

# Step 11 SHAP summary
SHAP_SUMMARY_FILE = (
    RESULTS_DIR / "step11_shap" /
    "step11_shap_summary.txt"
)

# ============================================================
# 2. APPROVED MODEL FEATURES
# ============================================================

APPROVED_FEATURES = [
    "storage_day",
    "temperature_c",
    "relative_humidity_pct",
    "weight_g",
    "weight_loss_pct",
    "days_since_first_observation",
    "observation_gap_days",
    "weight_change_g",
    "weight_loss_change_pct",
    "weight_loss_rate_pct_per_day",
    "temperature_change_c",
    "humidity_change_pct",
    "peel_color",
    "firmness"
]

EXPECTED_RISK_CLASSES = [
    "FRESH",
    "LOW RISK",
    "MODERATE RISK",
    "HIGH RISK",
    "VERY HIGH RISK"
]

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_file(path, description):

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

    print(f"[FOUND] {description}: {path}")


def load_model_artifact(path, description):

    print(f"\nLoading {description}...")

    artifact = joblib.load(path)

    if isinstance(artifact, dict):

        if "model" not in artifact:
            raise KeyError(
                f"{description} dictionary has no 'model' key."
            )

        model = artifact["model"]

        feature_columns = artifact.get(
            "feature_columns"
        )

        label_encoder = artifact.get(
            "label_encoder"
        )

        classes = artifact.get(
            "classes"
        )

    else:

        model = artifact
        feature_columns = None
        label_encoder = None
        classes = None

    print("Artifact type:", type(artifact))
    print("Model type:", type(model))

    return {
        "artifact": artifact,
        "model": model,
        "feature_columns": feature_columns,
        "label_encoder": label_encoder,
        "classes": classes
    }


def determine_feature_columns(model_info, model_name):

    feature_columns = model_info["feature_columns"]
    model = model_info["model"]

    if feature_columns is not None:

        feature_columns = list(feature_columns)

    elif hasattr(model, "feature_names_in_"):

        feature_columns = list(
            model.feature_names_in_
        )

    else:

        # Use the approved feature order only when
        # the saved artifact has no stored feature list.
        feature_columns = APPROVED_FEATURES.copy()

        print(
            f"NOTICE: {model_name} has no saved feature list. "
            "Using the approved Step 10C feature order."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            f"{model_name} contains duplicate feature names."
        )

    missing = [
        feature for feature in feature_columns
        if feature not in APPROVED_FEATURES
    ]

    if missing:
        raise ValueError(
            f"{model_name} contains unexpected predictors: {missing}"
        )

    return feature_columns


def validate_model_features(data, features, model_name):

    missing = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} is missing required input columns: {missing}"
        )


def predict_with_model(model, X, model_name):

    print(f"Running {model_name}...")

    predictions = model.predict(X)

    predictions = np.asarray(predictions).reshape(-1)

    if len(predictions) != len(X):
        raise ValueError(
            f"{model_name} returned an unexpected number of predictions."
        )

    return predictions


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print_section(
    "PHYSITWIN-X STEP 12 - DIGITAL TWIN INTEGRATION"
)

check_file(
    SHELF_LIFE_MODEL_FILE,
    "Shelf-life regression model"
)

check_file(
    RISK_MODEL_FILE,
    "Spoilage-risk classification model"
)

check_file(
    DYNAMIC_STATE_FILE,
    "Dynamic mango-state dataset"
)

if SHAP_SUMMARY_FILE.exists():

    print("[FOUND] Step 11 SHAP summary")

else:

    print(
        "[NOTICE] Step 11 SHAP summary not found. "
        "Predictions can still be integrated."
    )

# ============================================================
# 5. LOAD TRAINED MODELS
# ============================================================

print_section("LOADING TRAINED MODELS")

shelf_life_info = load_model_artifact(
    SHELF_LIFE_MODEL_FILE,
    "Step 7 shelf-life model"
)

risk_info = load_model_artifact(
    RISK_MODEL_FILE,
    "Step 10C spoilage-risk model"
)

shelf_life_model = shelf_life_info["model"]

risk_model = risk_info["model"]

risk_encoder = risk_info["label_encoder"]

# ============================================================
# 6. IDENTIFY MODEL FEATURE ORDERS
# ============================================================

print_section("VERIFYING MODEL FEATURE REQUIREMENTS")

shelf_life_features = determine_feature_columns(
    shelf_life_info,
    "Shelf-life regression model"
)

risk_features = determine_feature_columns(
    risk_info,
    "Spoilage-risk classification model"
)

print("\nSHELF-LIFE MODEL FEATURES:")

for index, feature in enumerate(
    shelf_life_features,
    start=1
):

    print(f"{index}. {feature}")

print("\nSPOILAGE-RISK MODEL FEATURES:")

for index, feature in enumerate(
    risk_features,
    start=1
):

    print(f"{index}. {feature}")

# ============================================================
# 7. VERIFY CLASS ENCODER
# ============================================================

print_section("VERIFYING SPOILAGE-RISK CLASSES")

if risk_encoder is None:

    raise ValueError(
        "The saved Step 10C model has no label encoder."
    )

encoder_classes = [
    str(value).strip().upper()
    for value in risk_encoder.classes_
]

print("Saved encoder classes:")

for index, class_name in enumerate(encoder_classes):

    print(f"{index}: {class_name}")

if set(encoder_classes) != set(EXPECTED_RISK_CLASSES):

    raise ValueError(
        "The saved risk classes do not match the five "
        "approved spoilage-risk categories."
    )

# ============================================================
# 8. LOAD DYNAMIC STATE DATASET
# ============================================================

print_section("LOADING DYNAMIC MANGO STATES")

data = pd.read_csv(DYNAMIC_STATE_FILE)

if data.empty:
    raise ValueError("Dynamic-state dataset is empty.")

required_keys = [
    "mango_id",
    "storage_day"
]

for column in required_keys:

    if column not in data.columns:
        raise ValueError(
            f"Missing required identifier: {column}"
        )

data["mango_id"] = (
    data["mango_id"]
    .astype(str)
    .str.strip()
)

data["storage_day"] = pd.to_numeric(
    data["storage_day"],
    errors="coerce"
)

if data[required_keys].isna().any().any():

    raise ValueError(
        "Missing mango ID or storage day detected."
    )

if data.duplicated(
    subset=["mango_id", "storage_day"]
).any():

    raise ValueError(
        "Duplicate mango/day records detected."
    )

data = data.sort_values(
    ["mango_id", "storage_day"]
).reset_index(drop=True)

print("Observations:", len(data))
print("Unique mangoes:", data["mango_id"].nunique())

print(
    "Storage-day range:",
    int(data["storage_day"].min()),
    "to",
    int(data["storage_day"].max())
)

# ============================================================
# 9. VERIFY REQUIRED PREDICTORS
# ============================================================

print_section("VERIFYING INPUT PREDICTORS")

validate_model_features(
    data,
    shelf_life_features,
    "Shelf-life model"
)

validate_model_features(
    data,
    risk_features,
    "Spoilage-risk model"
)

print("Shelf-life model inputs: PASSED")
print("Risk classifier inputs: PASSED")

# ============================================================
# 10. PREPARE MODEL INPUTS
# ============================================================

print_section("PREPARING DIGITAL TWIN INPUTS")

X_shelf_life = data[
    shelf_life_features
].copy()

X_risk = data[
    risk_features
].copy()

# Preserve missing first-observation lag values.
# The trained model pipelines perform their own
# preprocessing and imputation.

print("Shelf-life input shape:", X_shelf_life.shape)
print("Risk-classification input shape:", X_risk.shape)

# ============================================================
# 11. SHELF-LIFE PREDICTION
# ============================================================

print_section("STEP 7 - REMAINING SHELF-LIFE PREDICTION")

shelf_predictions = predict_with_model(
    shelf_life_model,
    X_shelf_life,
    "Shelf-life regression model"
)

data["predicted_remaining_shelf_life_days"] = (
    shelf_predictions
)

print(
    "Mean predicted remaining shelf life:",
    round(float(np.mean(shelf_predictions)), 4),
    "days"
)

print(
    "Minimum predicted remaining shelf life:",
    round(float(np.min(shelf_predictions)), 4),
    "days"
)

print(
    "Maximum predicted remaining shelf life:",
    round(float(np.max(shelf_predictions)), 4),
    "days"
)

# Keep raw predictions unchanged.
# Negative predictions, if any, are flagged for review.
data["shelf_life_prediction_below_zero_flag"] = (
    data["predicted_remaining_shelf_life_days"] < 0
)

# ============================================================
# 12. SPOILAGE-RISK CLASSIFICATION
# ============================================================

print_section("STEP 10C - SPOILAGE-RISK CLASSIFICATION")

risk_predictions_encoded = predict_with_model(
    risk_model,
    X_risk,
    "Spoilage-risk classifier"
)

risk_predictions_encoded = (
    risk_predictions_encoded.astype(int)
)

risk_predictions = risk_encoder.inverse_transform(
    risk_predictions_encoded
)

risk_predictions = [
    str(value).strip().upper()
    for value in risk_predictions
]

data["predicted_spoilage_risk"] = risk_predictions

# ============================================================
# 13. CLASS PROBABILITIES
# ============================================================

print_section("CALCULATING RISK CLASS PROBABILITIES")

if not hasattr(risk_model, "predict_proba"):

    raise AttributeError(
        "Risk classifier does not support predict_proba."
    )

risk_probabilities = risk_model.predict_proba(
    X_risk
)

risk_probabilities = np.asarray(
    risk_probabilities
)

classifier_class_names = [
    str(value).strip().upper()
    for value in risk_encoder.inverse_transform(
        np.asarray(risk_model.classes_, dtype=int)
    )
]

if risk_probabilities.shape != (
    len(data),
    len(classifier_class_names)
):

    raise ValueError(
        "Risk probability dimensions do not match "
        "the observations and model classes."
    )

for index, class_name in enumerate(
    classifier_class_names
):

    safe_name = (
        class_name.lower().replace(" ", "_")
    )

    data[
        f"probability_{safe_name}"
    ] = risk_probabilities[:, index]

data["maximum_risk_class_probability"] = (
    risk_probabilities.max(axis=1)
)

# ============================================================
# 14. INTEGRATED DIGITAL TWIN STATUS
# ============================================================

print_section("GENERATING INTEGRATED DIGITAL TWIN STATUS")

def determine_twin_status(row):

    risk = row["predicted_spoilage_risk"]

    shelf_life = row[
        "predicted_remaining_shelf_life_days"
    ]

    if risk == "VERY HIGH RISK":

        return "VERY HIGH RISK"

    elif risk == "HIGH RISK":

        return "HIGH RISK"

    elif risk == "MODERATE RISK":

        return "MODERATE RISK"

    elif risk == "LOW RISK":

        return "LOW RISK"

    elif risk == "FRESH":

        return "FRESH"

    return "UNCLASSIFIED"


data["digital_twin_status"] = data.apply(
    determine_twin_status,
    axis=1
)

# ============================================================
# 15. DIGITAL TWIN RECORD IDENTIFICATION
# ============================================================

data["digital_twin_record_id"] = (
    data["mango_id"].astype(str)
    + "_DAY_"
    + data["storage_day"].astype(int).astype(str)
)

# ============================================================
# 16. CREATE INTEGRATED OUTPUT
# ============================================================

print_section("SAVING INTEGRATED DIGITAL TWIN DATA")

integrated_output = (
    INTEGRATION_DIR /
    "physitwin_x_integrated_predictions.csv"
)

data.to_csv(
    integrated_output,
    index=False
)

print("Integrated dataset saved:")
print(integrated_output)

# ============================================================
# 17. CREATE LATEST OBSERVATION PER MANGO
# ============================================================

print_section("CREATING INDIVIDUAL MANGO DIGITAL TWIN RECORDS")

latest_records = (
    data.sort_values(
        ["mango_id", "storage_day"]
    )
    .groupby("mango_id", as_index=False)
    .tail(1)
    .sort_values("mango_id")
)

latest_output = (
    INTEGRATION_DIR /
    "physitwin_x_latest_mango_status.csv"
)

latest_records.to_csv(
    latest_output,
    index=False
)

print("Latest mango records:", len(latest_records))
print("Saved:", latest_output)

# ============================================================
# 18. CREATE RISK DISTRIBUTION SUMMARY
# ============================================================

print_section("SPOILAGE-RISK DISTRIBUTION")

risk_distribution = (
    data["predicted_spoilage_risk"]
    .value_counts()
    .reindex(
        EXPECTED_RISK_CLASSES,
        fill_value=0
    )
    .rename_axis("predicted_risk")
    .reset_index(name="observations")
)

risk_distribution["percentage"] = (
    risk_distribution["observations"]
    / len(data)
    * 100
)

risk_distribution.to_csv(
    INTEGRATION_DIR /
    "step12_risk_distribution.csv",
    index=False
)

print(
    risk_distribution.to_string(index=False)
)

# ============================================================
# 19. CREATE DAILY DIGITAL TWIN SUMMARY
# ============================================================

print_section("DAILY DIGITAL TWIN SUMMARY")

daily_summary = (
    data.groupby("storage_day")
    .agg(
        observation_count=("mango_id", "count"),
        mango_count=("mango_id", "nunique"),
        mean_predicted_shelf_life=(
            "predicted_remaining_shelf_life_days",
            "mean"
        ),
        mean_risk_probability=(
            "maximum_risk_class_probability",
            "mean"
        )
    )
    .reset_index()
)

daily_summary.to_csv(
    INTEGRATION_DIR /
    "step12_daily_digital_twin_summary.csv",
    index=False
)

print(daily_summary.round(4).to_string(index=False))

# ============================================================
# 20. VALIDATION CHECKS
# ============================================================

print_section("STEP 12 VALIDATION")

checks = []

checks.append((
    "Integrated dataset preserves observation count",
    len(data) == len(pd.read_csv(DYNAMIC_STATE_FILE))
))

checks.append((
    "All mango identifiers preserved",
    data["mango_id"].nunique() == 60
))

checks.append((
    "No duplicate mango/day records",
    not data.duplicated(
        ["mango_id", "storage_day"]
    ).any()
))

checks.append((
    "Shelf-life predictions contain no missing values",
    data[
        "predicted_remaining_shelf_life_days"
    ].notna().all()
))

checks.append((
    "Risk predictions contain no missing values",
    data["predicted_spoilage_risk"].notna().all()
))

checks.append((
    "All predicted classes are valid",
    set(data["predicted_spoilage_risk"].unique())
    .issubset(set(EXPECTED_RISK_CLASSES))
))

checks.append((
    "Risk probabilities are finite",
    np.isfinite(risk_probabilities).all()
))

checks.append((
    "Risk probabilities sum approximately to one",
    np.allclose(
        risk_probabilities.sum(axis=1),
        1.0,
        atol=1e-5
    )
))

checks.append((
    "Integrated CSV exists",
    integrated_output.exists()
))

checks.append((
    "Latest mango status CSV exists",
    latest_output.exists()
))

passed = 0
failed = 0

for description, result in checks:

    if result:

        print("[PASS]", description)
        passed += 1

    else:

        print("[FAIL]", description)
        failed += 1

print("\nVALIDATION TOTAL")
print("Passed:", passed)
print("Failed:", failed)

# ============================================================
# 21. WRITE INTEGRATION REPORT
# ============================================================

report_file = (
    INTEGRATION_DIR /
    "step12_integration_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHYSITWIN-X STEP 12\n"
        "DIGITAL TWIN INTEGRATION REPORT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write("DATASET INFORMATION\n")
    file.write(f"Observations: {len(data)}\n")
    file.write(f"Mangoes: {data['mango_id'].nunique()}\n")
    file.write(
        f"Storage days: {data['storage_day'].min()} "
        f"to {data['storage_day'].max()}\n\n"
    )

    file.write("INTEGRATED COMPONENTS\n")
    file.write("- Step 7 shelf-life regression model\n")
    file.write("- Step 9 dynamic-state dataset\n")
    file.write("- Step 10C spoilage-risk classifier\n")
    file.write("- Step 11 SHAP explainability reference\n\n")

    file.write("SHELF-LIFE PREDICTION SUMMARY\n")
    file.write(
        f"Mean: {data['predicted_remaining_shelf_life_days'].mean():.4f} days\n"
    )

    file.write(
        f"Minimum: {data['predicted_remaining_shelf_life_days'].min():.4f} days\n"
    )

    file.write(
        f"Maximum: {data['predicted_remaining_shelf_life_days'].max():.4f} days\n\n"
    )

    file.write("RISK CLASS DISTRIBUTION\n")
    file.write(
        risk_distribution.to_string(index=False)
    )

    file.write("\n\nVALIDATION RESULTS\n")

    for description, result in checks:

        file.write(
            f"{'PASS' if result else 'FAIL'}: "
            f"{description}\n"
        )

    file.write(
        f"\nTOTAL PASSED: {passed}\n"
    )

    file.write(
        f"TOTAL FAILED: {failed}\n"
    )

    file.write("\nINTERPRETATION LIMITATIONS\n")

    file.write(
        "The integrated predictions are generated from the "
        "existing trained models and recorded observations.\n"
    )

    file.write(
        "This integration run is not an independent model "
        "performance evaluation.\n"
    )

    file.write(
        "The recorded spoilage-risk labels may depend on "
        "firmness and peel-color characteristics used as predictors.\n"
    )

    file.write(
        "SHAP values explain model behavior and do not establish "
        "causation.\n"
    )

    file.write(
        "The existing dataset does not establish live sensor "
        "connectivity or automatic real-time updates.\n"
    )

# ============================================================
# 22. FINAL RESULT
# ============================================================

print_section("PHYSITWIN-X STEP 12 FINAL RESULT")

print("Validation passed:", passed)
print("Validation failed:", failed)

print("\nOUTPUT DIRECTORY:")
print(INTEGRATION_DIR)

print("\nGENERATED FILES:")
print("- physitwin_x_integrated_predictions.csv")
print("- physitwin_x_latest_mango_status.csv")
print("- step12_risk_distribution.csv")
print("- step12_daily_digital_twin_summary.csv")
print("- step12_integration_report.txt")

if failed == 0:

    print("\nSTEP 12 INTEGRATION VALIDATION PASSED.")

else:

    print(
        "\nSTEP 12 GENERATED OUTPUTS, BUT "
        "SOME VALIDATION CHECKS FAILED."
    )

    raise RuntimeError(
        "Review the failed validation checks before proceeding."
    )

print("\nIMPORTANT:")
print(
    "This is an integration run using the existing "
    "745 recorded observations."
)

print(
    "It is not yet a live, sensor-connected digital twin."
)

print("=" * 70)
