# ============================================================
# PHYSITWIN-X STEP 10D
# SPOILAGE-RISK MODEL VALIDATION AND LABEL-DEPENDENCY ASSESSMENT
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "physitwin_x_risk_train.csv"
TEST_FILE = DATA / "physitwin_x_risk_test.csv"

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("=" * 65)
print("PHYSITWIN-X STEP 10D")
print("LABEL-DEPENDENCY ASSESSMENT")
print("=" * 65)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

TARGET = "spoilage_risk"
GROUP = "mango_id"

if TARGET not in train.columns or TARGET not in test.columns:
    raise ValueError("The spoilage_risk target column is missing.")

if GROUP not in train.columns or GROUP not in test.columns:
    raise ValueError("The mango_id column is missing.")

print("\nDATASET INFORMATION")
print("Training observations:", len(train))
print("Training mangoes:", train[GROUP].nunique())
print("Testing observations:", len(test))
print("Testing mangoes:", test[GROUP].nunique())

# Confirm no mango appears in both sets.
overlap = set(train[GROUP]) & set(test[GROUP])

if overlap:
    raise ValueError(
        f"Data leakage detected: shared mango IDs: {sorted(overlap)}"
    )

print("Mango-level separation: PASSED")

# ------------------------------------------------------------
# 3. DEFINE APPROVED PREDICTORS
# ------------------------------------------------------------

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

FULL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REDUCED_FEATURES = [
    feature for feature in FULL_FEATURES
    if feature not in ["peel_color", "firmness"]
]

DAY_ONLY_FEATURES = ["storage_day"]

required_columns = (
    FULL_FEATURES + [TARGET, GROUP]
)

missing_columns = [
    column for column in required_columns
    if column not in train.columns or column not in test.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 4. ENCODE TARGET LABELS
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train[TARGET].astype(str)
)

y_test = label_encoder.transform(
    test[TARGET].astype(str)
)

class_names = label_encoder.classes_
number_of_classes = len(class_names)

if number_of_classes != 5:
    raise ValueError(
        f"Expected 5 risk classes, found {number_of_classes}."
    )

print("\nRISK CLASSES")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

# Balanced sample weights, calculated from training data only.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# ------------------------------------------------------------
# 5. MODEL CONFIGURATIONS
# ------------------------------------------------------------

def create_preprocessor(features):

    numeric = [
        feature for feature in NUMERIC_FEATURES
        if feature in features
    ]

    categorical = [
        feature for feature in CATEGORICAL_FEATURES
        if feature in features
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def create_full_or_reduced_model(features):

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(features)),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=number_of_classes,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        ))
    ])

    return pipeline


def create_day_only_model():

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(
            DAY_ONLY_FEATURES
        )),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=number_of_classes,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        ))
    ])

    return pipeline


# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE EACH MODEL
# ------------------------------------------------------------

model_definitions = {
    "FULL_FEATURE": FULL_FEATURES,
    "WITHOUT_FIRMNESS_AND_PEEL": REDUCED_FEATURES,
    "STORAGE_DAY_ONLY": DAY_ONLY_FEATURES
}

metrics_rows = []
prediction_tables = []

for model_name, features in model_definitions.items():

    print("\n" + "=" * 65)
    print("TRAINING:", model_name)
    print("=" * 65)

    X_train = train[features].copy()
    X_test = test[features].copy()

    if model_name == "STORAGE_DAY_ONLY":
        model = create_day_only_model()
    else:
        model = create_full_or_reduced_model(features)

    model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )
    macro_precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    macro_recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )

    metrics_rows.append({
        "model": model_name,
        "predictor_count": len(features),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    prediction_table = pd.DataFrame({
        "mango_id": test[GROUP].values,
        "storage_day": test["storage_day"].values,
        "actual_risk": label_encoder.inverse_transform(y_test),
        "predicted_risk": label_encoder.inverse_transform(
            predictions
        ),
        "correct": predictions == y_test,
        "model": model_name
    })

    prediction_tables.append(prediction_table)

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(number_of_classes),
        target_names=class_names,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(number_of_classes)
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    )

    matrix_df.to_csv(
        RESULTS / f"step10D_{model_name.lower()}_confusion_matrix.csv"
    )

    with open(
        RESULTS / f"step10D_{model_name.lower()}_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"MODEL: {model_name}\n\n")
        file.write(f"Predictors: {features}\n\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(matrix_df.to_string())

    print("Accuracy:", round(accuracy, 4))
    print("Balanced accuracy:", round(balanced_accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

# ------------------------------------------------------------
# 7. SAVE COMPARISON RESULTS
# ------------------------------------------------------------

metrics_df = pd.DataFrame(metrics_rows)

metrics_df.to_csv(
    RESULTS / "step10D_model_comparison.csv",
    index=False
)

all_predictions = pd.concat(
    prediction_tables,
    ignore_index=True
)

all_predictions.to_csv(
    RESULTS / "step10D_model_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 8. CALCULATE PERFORMANCE DIFFERENCES
# ------------------------------------------------------------

full_accuracy = metrics_df.loc[
    metrics_df["model"] == "FULL_FEATURE",
    "accuracy"
].iloc[0]

reduced_accuracy = metrics_df.loc[
    metrics_df["model"] == "WITHOUT_FIRMNESS_AND_PEEL",
    "accuracy"
].iloc[0]

day_accuracy = metrics_df.loc[
    metrics_df["model"] == "STORAGE_DAY_ONLY",
    "accuracy"
].iloc[0]

accuracy_difference = full_accuracy - reduced_accuracy
day_difference = full_accuracy - day_accuracy

# ------------------------------------------------------------
# 9. FINAL REPORT
# ------------------------------------------------------------

report_path = RESULTS / "step10D_validation_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("PHYSITWIN-X STEP 10D\n")
    file.write("SPOILAGE-RISK MODEL VALIDATION\n")
    file.write("=" * 60 + "\n\n")

    file.write("DATASET\n")
    file.write(f"Training observations: {len(train)}\n")
    file.write(f"Testing observations: {len(test)}\n")
    file.write(f"Training mangoes: {train[GROUP].nunique()}\n")
    file.write(f"Testing mangoes: {test[GROUP].nunique()}\n\n")

    file.write("MODEL COMPARISON\n")
    file.write(metrics_df.to_string(index=False))
    file.write("\n\n")

    file.write("ACCURACY DIFFERENCES\n")
    file.write(
        f"Full model minus reduced model: "
        f"{accuracy_difference:.4f}\n"
    )
    file.write(
        f"Full model minus day-only baseline: "
        f"{day_difference:.4f}\n\n"
    )

    file.write("INTERPRETATION NOTES\n")
    file.write(
        "The reduced model excludes firmness and peel color.\n"
    )
    file.write(
        "A performance decrease may indicate that these variables "
        "are strongly associated with the recorded risk labels.\n"
    )
    file.write(
        "The comparison does not by itself establish causation "
        "or independent future-spoilage prediction.\n"
    )
    file.write(
        "The test set contains only 12 mangoes; further "
        "mango-level cross-validation and external validation "
        "are recommended.\n"
    )

print("\n" + "=" * 65)
print("STEP 10D COMPLETED")
print("=" * 65)

print("\nMODEL COMPARISON")
print(metrics_df.round(4).to_string(index=False))

print("\nAccuracy difference: full vs reduced =",
      round(accuracy_difference, 4))

print("Accuracy difference: full vs day-only =",
      round(day_difference, 4))

print("\nFILES SAVED:")
print("results/step10D_model_comparison.csv")
print("results/step10D_model_predictions.csv")
print("results/step10D_validation_summary.txt")
print("Three model reports and confusion matrices.")

print("\nSTEP 10D FINISHED.")