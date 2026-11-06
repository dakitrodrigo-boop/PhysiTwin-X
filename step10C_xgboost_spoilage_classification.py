
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PHYSITWIN-X STEP 10C
# CORRECTED XGBOOST SPOILAGE-RISK CLASSIFICATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "physitwin_x_risk_train.csv"
TEST_FILE = DATA_DIR / "physitwin_x_risk_test.csv"

MODEL_FILE = RESULTS_DIR / "physitwin_x_spoilage_risk_model.joblib"
PREDICTIONS_FILE = RESULTS_DIR / "step10C_risk_predictions.csv"
BASELINE_FILE = RESULTS_DIR / "step10C_baseline_predictions.csv"
METRICS_FILE = RESULTS_DIR / "step10C_model_performance.csv"
CLASS_METRICS_FILE = RESULTS_DIR / "step10C_per_class_metrics.csv"
IMPORTANCE_FILE = RESULTS_DIR / "step10C_feature_importance.csv"
REPORT_FILE = RESULTS_DIR / "step10C_classification_report.txt"
CONFUSION_FILE = FIGURES_DIR / "step10C_confusion_matrix.png"

TARGET = "spoilage_risk"

CLASSES = [
    "FRESH",
    "LOW RISK",
    "MODERATE RISK",
    "HIGH RISK",
    "VERY HIGH RISK"
]

NUMERIC_FEATURES = [
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
    "humidity_change_pct"
]

CATEGORICAL_FEATURES = [
    "peel_color",
    "firmness"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("PHYSITWIN-X STEP 10C")
print("CORRECTED XGBOOST SPOILAGE-RISK CLASSIFICATION")
print("=" * 65)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("\nDATASET INFORMATION")
print("-" * 65)

print(f"Training observations: {len(train)}")
print(f"Training mangoes: {train['mango_id'].nunique()}")
print(f"Testing observations: {len(test)}")
print(f"Testing mangoes: {test['mango_id'].nunique()}")

# ============================================================
# DATA VALIDATION
# ============================================================

assert train.columns.is_unique
assert test.columns.is_unique

assert set(ALL_FEATURES + [TARGET]).issubset(train.columns)
assert set(ALL_FEATURES + [TARGET]).issubset(test.columns)

assert train[TARGET].notna().all()
assert test[TARGET].notna().all()

assert set(train[TARGET].unique()) == set(CLASSES)
assert set(test[TARGET].unique()) == set(CLASSES)

assert set(train["mango_id"]).isdisjoint(
    set(test["mango_id"])
)

print("\nDATA VALIDATION: PASSED")

# ============================================================
# PREPARE FEATURES AND ENCODE TARGET LABELS
# ============================================================

X_train = train[ALL_FEATURES].copy()
X_test = test[ALL_FEATURES].copy()

y_train = train[TARGET].copy()
y_test = test[TARGET].copy()

# Convert text classes into integer labels.
# Original text labels are preserved for reporting.

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)

y_test_encoded = label_encoder.transform(y_test)

print("\nCLASS LABEL ENCODING")
print("-" * 65)

for index, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} = {index}")

# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)

# ============================================================
# CLASS IMBALANCE HANDLING
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded
)

print("\nTRAINING CLASS DISTRIBUTION")
print("-" * 65)
print(y_train.value_counts().to_string())

# ============================================================
# MODEL 1: FULL-FEATURE XGBOOST CLASSIFIER
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FULL-FEATURE XGBOOST CLASSIFIER")
print("=" * 65)

full_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=5,
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

full_model.fit(
    X_train,
    y_train_encoded,
    classifier__sample_weight=sample_weights
)

predictions_encoded = full_model.predict(X_test)

predictions = label_encoder.inverse_transform(
    predictions_encoded.astype(int)
)

# ============================================================
# FULL MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

macro_precision = precision_score(
    y_test,
    predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

# ============================================================
# MODEL 2: STORAGE-DAY-ONLY BASELINE
# ============================================================

print("\n" + "=" * 65)
print("TRAINING STORAGE-DAY-ONLY BASELINE")
print("=" * 65)

baseline_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=5,
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=1.0,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

baseline_model.fit(
    train[["storage_day"]],
    y_train_encoded,
    classifier__sample_weight=sample_weights
)

baseline_predictions_encoded = baseline_model.predict(
    test[["storage_day"]]
)

baseline_predictions = label_encoder.inverse_transform(
    baseline_predictions_encoded.astype(int)
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_predictions
)

baseline_macro_f1 = f1_score(
    y_test,
    baseline_predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

# ============================================================
# SAVE FULL MODEL PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame({
    "mango_id": test["mango_id"].values,
    "storage_day": test["storage_day"].values,
    "actual_spoilage_risk": y_test.values,
    "predicted_spoilage_risk": predictions,
    "correct_prediction": (
        y_test.values == predictions
    )
})

prediction_output.to_csv(
    PREDICTIONS_FILE,
    index=False
)

# ============================================================
# SAVE BASELINE PREDICTIONS
# ============================================================

baseline_output = pd.DataFrame({
    "mango_id": test["mango_id"].values,
    "storage_day": test["storage_day"].values,
    "actual_spoilage_risk": y_test.values,
    "baseline_predicted_risk": baseline_predictions,
    "correct_prediction": (
        y_test.values == baseline_predictions
    )
})

baseline_output.to_csv(
    BASELINE_FILE,
    index=False
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

classification_text = classification_report(
    y_test,
    predictions,
    labels=CLASSES,
    zero_division=0
)

report_dict = classification_report(
    y_test,
    predictions,
    labels=CLASSES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
    CLASS_METRICS_FILE
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=CLASSES
)

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(
    cm,
    cmap="Blues"
)

ax.set_title(
    "PhysiTwin-X Spoilage-Risk Classification\nConfusion Matrix"
)

ax.set_xlabel("Predicted Risk Class")
ax.set_ylabel("Actual Risk Class")

ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))

ax.set_xticklabels(
    CLASSES,
    rotation=35,
    ha="right"
)

ax.set_yticklabels(CLASSES)

for row in range(len(CLASSES)):
    for col in range(len(CLASSES)):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax)

fig.tight_layout()

fig.savefig(
    CONFUSION_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor = full_model.named_steps[
    "preprocessor"
]

fitted_classifier = full_model.named_steps[
    "classifier"
]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_values = fitted_classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

metrics = pd.DataFrame([
    {
        "model": "Full-feature XGBoost",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    },
    {
        "model": "Storage-day-only baseline",
        "accuracy": baseline_accuracy,
        "balanced_accuracy": baseline_balanced_accuracy,
        "macro_precision": np.nan,
        "macro_recall": np.nan,
        "macro_f1": baseline_macro_f1,
        "weighted_f1": np.nan
    }
])

metrics.to_csv(
    METRICS_FILE,
    index=False
)

# ============================================================
# SAVE TRAINED MODEL AND LABEL ENCODER
# ============================================================

joblib.dump(
    {
        "model": full_model,
        "label_encoder": label_encoder,
        "feature_columns": ALL_FEATURES,
        "classes": CLASSES
    },
    MODEL_FILE
)

# ============================================================
# FINAL REPORT
# ============================================================

report_lines = []

def log(message=""):
    print(message)
    report_lines.append(str(message))

log("\n" + "=" * 65)
log("PHYSITWIN-X STEP 10C RESULTS")
log("=" * 65)

log("\nFULL-FEATURE XGBOOST PERFORMANCE")

log(f"Accuracy: {accuracy:.4f}")
log(f"Balanced accuracy: {balanced_accuracy:.4f}")
log(f"Macro precision: {macro_precision:.4f}")
log(f"Macro recall: {macro_recall:.4f}")
log(f"Macro F1-score: {macro_f1:.4f}")
log(f"Weighted F1-score: {weighted_f1:.4f}")

log("\nSTORAGE-DAY-ONLY BASELINE")

log(f"Accuracy: {baseline_accuracy:.4f}")
log(f"Balanced accuracy: {baseline_balanced_accuracy:.4f}")
log(f"Macro F1-score: {baseline_macro_f1:.4f}")

log("\nPER-CLASS CLASSIFICATION REPORT")
log(classification_text)

log("\nTOP 10 FEATURES")
log(
    importance_df.head(10).to_string(index=False)
)

log("\nCONFUSION MATRIX")
log(str(cm))

log("\nOUTPUT FILES")

log(str(MODEL_FILE))
log(str(PREDICTIONS_FILE))
log(str(BASELINE_FILE))
log(str(METRICS_FILE))
log(str(CLASS_METRICS_FILE))
log(str(IMPORTANCE_FILE))
log(str(CONFUSION_FILE))

log("\nIMPORTANT RESEARCH LIMITATIONS")

log(
    "This model classifies recorded spoilage-risk labels. "
    "It does not independently establish future spoilage."
)

log(
    "The test set contains observations from only 12 mangoes. "
    "Further grouped cross-validation and external validation "
    "are needed before claiming general performance."
)

log(
    "The model's feature importance indicates predictive use, "
    "not causal influence."
)

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(f"\nReport saved to: {REPORT_FILE}")

print("\nSTEP 10C COMPLETED")
