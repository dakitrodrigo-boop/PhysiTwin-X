
# ============================================================
# PHYSITWIN-X
# STEP 7: XGBOOST SHELF-LIFE PREDICTION MODEL
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "physitwin_x_train.csv"
TEST_FILE = DATA_DIR / "physitwin_x_test.csv"

PREDICTION_FILE = RESULTS_DIR / "step7_shelf_life_predictions.csv"
METRICS_FILE = RESULTS_DIR / "step7_model_performance.csv"
SUMMARY_FILE = RESULTS_DIR / "step7_model_summary.txt"
MODEL_FILE = RESULTS_DIR / "step7_xgboost_model.joblib"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "remaining_shelf_life_days"
GROUP = "mango_id"

RANDOM_STATE = 42

# ============================================================
# 2. DISPLAY PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("PHYSITWIN-X")
print("STEP 7: XGBOOST SHELF-LIFE PREDICTION")
print("=" * 70)

# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Testing file not found: {TEST_FILE}"
    )

# ============================================================
# 4. LOAD DATASETS
# ============================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

train.columns = (
    train.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

test.columns = (
    test.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print("\nDATASET INFORMATION")
print("Training observations:", len(train))
print("Testing observations:", len(test))
print("Training mangoes:", train[GROUP].nunique())
print("Testing mangoes:", test[GROUP].nunique())

# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [GROUP, TARGET]

for column in required_columns:
    if column not in train.columns:
        raise ValueError(
            f"Missing training column: {column}"
        )

    if column not in test.columns:
        raise ValueError(
            f"Missing testing column: {column}"
        )

if list(train.columns) != list(test.columns):
    raise ValueError(
        "Training and testing columns do not match."
    )

# ============================================================
# 6. CHECK MANGO-ID SEPARATION
# ============================================================

train_mangoes = set(train[GROUP].dropna().unique())
test_mangoes = set(test[GROUP].dropna().unique())

overlap = train_mangoes.intersection(test_mangoes)

if overlap:
    raise ValueError(
        f"Data leakage detected. Overlapping mango IDs: {overlap}"
    )

print("\nMANGO-ID SEPARATION")
print("Overlapping mango IDs:", len(overlap))
print("Leakage check: PASSED")

# ============================================================
# 7. VALIDATE TARGET VALUES
# ============================================================

train[TARGET] = pd.to_numeric(
    train[TARGET], errors="coerce"
)

test[TARGET] = pd.to_numeric(
    test[TARGET], errors="coerce"
)

train = train.dropna(subset=[GROUP, TARGET]).copy()
test = test.dropna(subset=[GROUP, TARGET]).copy()

if train.empty or test.empty:
    raise ValueError("Training or testing data is empty.")

if (train[TARGET] < 0).any():
    raise ValueError("Negative training targets found.")

if (test[TARGET] < 0).any():
    raise ValueError("Negative testing targets found.")

# ============================================================
# 8. SELECT LEAKAGE-CONTROLLED FEATURES
# ============================================================

excluded_columns = {
    GROUP,
    TARGET,
    "endpoint_day",
    "spoilage_risk",
    "ripeness_stage",
    "post_endpoint_observation",
    "shelf_life_training_eligible",
    "data_review_required",
}

features = [
    column for column in train.columns
    if column not in excluded_columns
]

if not features:
    raise ValueError("No valid predictor features found.")

# Check that test data has exactly the same predictors.
X_train = train[features].copy()
X_test = test[features].copy()

y_train = train[TARGET].copy()
y_test = test[TARGET].copy()

# ============================================================
# 9. IDENTIFY NUMERICAL FEATURES
# ============================================================

numeric_candidates = [
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
]

for column in numeric_candidates:
    if column in features:
        X_train[column] = pd.to_numeric(
            X_train[column], errors="coerce"
        )

        X_test[column] = pd.to_numeric(
            X_test[column], errors="coerce"
        )

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    column for column in features
    if column not in numeric_features
]

print("\nSELECTED PREDICTORS")

for feature in features:
    print("-", feature)

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)

# ============================================================
# 10. BUILD PREPROCESSING PIPELINE
# ============================================================

transformers = []

if numeric_features:

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    transformers.append((
        "numeric",
        numeric_pipeline,
        numeric_features
    ))

if categorical_features:

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers.append((
        "categorical",
        categorical_pipeline,
        categorical_features
    ))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

# ============================================================
# 11. INITIALIZE XGBOOST REGRESSION
# ============================================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="rmse"
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgboost", xgb_model)
])

print("\nMODEL CONFIGURATION")
print("Algorithm: XGBoost Regression")
print("Estimators: 300")
print("Maximum depth: 3")
print("Learning rate: 0.03")

# ============================================================
# 12. TRAIN XGBOOST MODEL
# ============================================================

print("\nTRAINING XGBOOST MODEL...")

model_pipeline.fit(X_train, y_train)

print("XGBOOST TRAINING COMPLETED.")

# ============================================================
# 13. GENERATE TEST PREDICTIONS
# ============================================================

print("\nGENERATING TEST PREDICTIONS...")

predictions = model_pipeline.predict(X_test)

# Keep raw model predictions for evaluation.
# Also create nonnegative operational estimates.
operational_predictions = np.maximum(predictions, 0)

# ============================================================
# 14. TRAIN MEDIAN BASELINE
# ============================================================

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DummyRegressor(strategy="median"))
])

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

# ============================================================
# 15. CALCULATE PERFORMANCE METRICS
# ============================================================

def evaluate_model(actual, predicted):

    return {
        "MAE_days": mean_absolute_error(
            actual, predicted
        ),

        "RMSE_days": np.sqrt(
            mean_squared_error(actual, predicted)
        ),

        "R2": r2_score(
            actual, predicted
        )
    }


xgb_metrics = evaluate_model(
    y_test,
    predictions
)

baseline_metrics = evaluate_model(
    y_test,
    baseline_predictions
)

metrics = pd.DataFrame([
    {
        "model": "XGBoost Regression",
        **xgb_metrics
    },
    {
        "model": "Median Baseline",
        **baseline_metrics
    }
])

metrics.to_csv(METRICS_FILE, index=False)

# ============================================================
# 16. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({

    "mango_id": test[GROUP].values,

    "storage_day": (
        test["storage_day"].values
        if "storage_day" in test.columns
        else np.nan
    ),

    "actual_remaining_shelf_life_days":
        y_test.values,

    "predicted_remaining_shelf_life_days":
        predictions,

    "nonnegative_predicted_shelf_life_days":
        operational_predictions,

    "prediction_error_days":
        y_test.values - predictions,

    "absolute_error_days":
        np.abs(y_test.values - predictions),

    "baseline_prediction_days":
        baseline_predictions
})

prediction_results.to_csv(
    PREDICTION_FILE,
    index=False
)

# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model_pipeline,
    MODEL_FILE
)

# ============================================================
# 18. SAVE MODEL SUMMARY
# ============================================================

summary = f"""
PHYSITWIN-X
STEP 7: XGBOOST SHELF-LIFE PREDICTION

MODEL:
XGBoost Regression

TARGET:
{TARGET}

TRAINING OBSERVATIONS:
{len(X_train)}

TESTING OBSERVATIONS:
{len(X_test)}

TRAINING MANGOES:
{len(train_mangoes)}

TESTING MANGOES:
{len(test_mangoes)}

MANGO-ID OVERLAP:
{len(overlap)}

PREDICTION FEATURES:
{features}

XGBOOST PERFORMANCE:
MAE: {xgb_metrics['MAE_days']:.4f} days
RMSE: {xgb_metrics['RMSE_days']:.4f} days
R2: {xgb_metrics['R2']:.4f}

MEDIAN BASELINE PERFORMANCE:
MAE: {baseline_metrics['MAE_days']:.4f} days
RMSE: {baseline_metrics['RMSE_days']:.4f} days
R2: {baseline_metrics['R2']:.4f}

NEGATIVE RAW PREDICTIONS:
{int((predictions < 0).sum())}

NOTE:
The raw predictions are used for performance evaluation.
Nonnegative estimates are also saved for operational review.

The model was evaluated on held-out mango IDs.
Results are preliminary and based on 60 mangoes.

This step trains a shelf-life regression model.
It does not yet constitute the complete digital twin.
"""

SUMMARY_FILE.write_text(
    summary,
    encoding="utf-8"
)

# ============================================================
# 19. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST MODEL PERFORMANCE")
print("=" * 70)

print(metrics.to_string(index=False))

print("\nNEGATIVE RAW PREDICTIONS:",
      int((predictions < 0).sum()))

print("\nSAVED FILES")

print("Predictions:", PREDICTION_FILE)
print("Metrics:", METRICS_FILE)
print("Summary:", SUMMARY_FILE)
print("Trained model:", MODEL_FILE)

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY.")
print("=" * 70)
