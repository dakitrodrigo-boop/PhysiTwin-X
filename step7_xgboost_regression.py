
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# PHYSITWIN-X
# STEP 7: XGBOOST REGRESSION MODEL DEVELOPMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_FILE = DATA_DIR / "physitwin_x_train.csv"
TEST_FILE = DATA_DIR / "physitwin_x_test.csv"

MODEL_FILE = RESULTS_DIR / "physitwin_x_xgboost_model.joblib"
PREDICTIONS_FILE = RESULTS_DIR / "step7_shelf_life_predictions.csv"
METRICS_FILE = RESULTS_DIR / "step7_model_performance.txt"
IMPORTANCE_FILE = RESULTS_DIR / "step7_feature_importance.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "remaining_shelf_life_days"
GROUP = "mango_id"

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

NUMERIC_FEATURES = [
    col for col in FEATURES
    if col not in ["peel_color", "firmness"]
]

CATEGORICAL_FEATURES = [
    "peel_color",
    "firmness"
]


# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
# ============================================================

print("=" * 65)
print("PHYSITWIN-X STEP 7: XGBOOST REGRESSION")
print("=" * 65)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Missing file: {TEST_FILE}")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDATASET INFORMATION")
print("Training observations:", len(train_df))
print("Training mangoes:", train_df[GROUP].nunique())
print("Testing observations:", len(test_df))
print("Testing mangoes:", test_df[GROUP].nunique())


# ============================================================
# 2. VALIDATE INPUT DATA
# ============================================================

required_columns = FEATURES + [TARGET, GROUP]

for name, dataset in [
    ("Training", train_df),
    ("Testing", test_df)
]:
    missing = [
        col for col in required_columns
        if col not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

    dataset[TARGET] = pd.to_numeric(
        dataset[TARGET], errors="coerce"
    )

    if dataset[TARGET].isna().any():
        raise ValueError(
            f"{name} dataset has missing or invalid targets."
        )

    if (dataset[TARGET] < 0).any():
        raise ValueError(
            f"{name} dataset contains negative targets."
        )

train_groups = set(train_df[GROUP].astype(str))
test_groups = set(test_df[GROUP].astype(str))

if train_groups.intersection(test_groups):
    raise ValueError(
        "Data leakage detected: mangoes overlap between datasets."
    )

print("\nDATA VALIDATION")
print("Required columns: PASSED")
print("Target validation: PASSED")
print("Mango group separation: PASSED")


# ============================================================
# 3. PREPARE FEATURES AND TARGETS
# ============================================================

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()

for col in CATEGORICAL_FEATURES:
    X_train[col] = X_train[col].astype("string")
    X_test[col] = X_test[col].astype("string")

for col in NUMERIC_FEATURES:
    X_train[col] = pd.to_numeric(
        X_train[col], errors="coerce"
    )
    X_test[col] = pd.to_numeric(
        X_test[col], errors="coerce"
    )


# ============================================================
# 4. BUILD PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# 5. CONFIGURE XGBOOST REGRESSION
# ============================================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3,
    random_state=42,
    n_jobs=4,
    verbosity=0
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model)
    ]
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nMODEL TRAINING")
print("Algorithm: XGBoost Regression")
print("Predictors:", len(FEATURES))
print("Training observations:", len(X_train))

model_pipeline.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# 7. PREDICT UNSEEN TEST DATA
# ============================================================

print("\nMODEL PREDICTION")

y_pred = model_pipeline.predict(X_test)

# Shelf life cannot be negative.
# Preserve the raw model output separately for auditing.
y_pred_raw = y_pred.copy()
y_pred = np.maximum(y_pred, 0)

print("Testing predictions completed.")


# ============================================================
# 8. CALCULATE MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

# Baseline: always predict the training-set mean.
baseline_pred = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test, baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

baseline_r2 = r2_score(
    y_test, baseline_pred
)


# ============================================================
# 9. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "mango_id": test_df[GROUP].values,
    "storage_day": test_df["storage_day"].values,
    "actual_remaining_shelf_life_days": y_test.values,
    "predicted_remaining_shelf_life_days": y_pred,
    "raw_model_prediction_days": y_pred_raw,
    "absolute_error_days": np.abs(
        y_test.values - y_pred
    )
})

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 10. SAVE FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor = (
    model_pipeline.named_steps["preprocessor"]
)

fitted_regressor = (
    model_pipeline.named_steps["regressor"]
)

transformed_names = (
    fitted_preprocessor.get_feature_names_out()
)

importance = pd.DataFrame({
    "feature": transformed_names,
    "importance": fitted_regressor.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

joblib.dump(model_pipeline, MODEL_FILE)


# ============================================================
# 12. GENERATE PERFORMANCE REPORT
# ============================================================

report = [
    "=" * 65,
    "PHYSITWIN-X STEP 7 MODEL PERFORMANCE REPORT",
    "=" * 65,
    "",
    "MODEL INFORMATION",
    "Algorithm: XGBoost Regression",
    f"Number of predictors: {len(FEATURES)}",
    f"Training observations: {len(train_df)}",
    f"Training mangoes: {len(train_groups)}",
    f"Testing observations: {len(test_df)}",
    f"Testing mangoes: {len(test_groups)}",
    "",
    "XGBOOST TEST PERFORMANCE",
    f"MAE: {mae:.4f} days",
    f"RMSE: {rmse:.4f} days",
    f"R2: {r2:.4f}",
    "",
    "BASELINE TEST PERFORMANCE",
    f"Baseline MAE: {baseline_mae:.4f} days",
    f"Baseline RMSE: {baseline_rmse:.4f} days",
    f"Baseline R2: {baseline_r2:.4f}",
    "",
    "DATA LEAKAGE CHECK",
    "Shared mango IDs: 0",
    "Status: PASSED",
    "",
    "OUTPUT FILES",
    str(MODEL_FILE),
    str(PREDICTIONS_FILE),
    str(IMPORTANCE_FILE),
    "",
    "STEP 7 STATUS: COMPLETED"
]

METRICS_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("XGBOOST REGRESSION PERFORMANCE")
print("=" * 65)

print(f"MAE:  {mae:.4f} days")
print(f"RMSE: {rmse:.4f} days")
print(f"R2:   {r2:.4f}")

print("\nBASELINE PERFORMANCE")
print(f"MAE:  {baseline_mae:.4f} days")
print(f"RMSE: {baseline_rmse:.4f} days")
print(f"R2:   {baseline_r2:.4f}")

print("\nTOP 10 FEATURES")
print(importance.head(10).to_string(index=False))

print("\nOUTPUT FILES")
print("Model:", MODEL_FILE)
print("Predictions:", PREDICTIONS_FILE)
print("Feature importance:", IMPORTANCE_FILE)
print("Performance report:", METRICS_FILE)

print("\n" + "=" * 65)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 65)
