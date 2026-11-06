
# ============================================================
# PHYSITWIN-X STEP 10E
# MANGO-LEVEL GROUP CROSS-VALIDATION
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "physitwin_x_risk_train.csv"

# ------------------------------------------------------------
# 2. LOAD TRAINING DATA ONLY
# ------------------------------------------------------------

print("=" * 65)
print("PHYSITWIN-X STEP 10E")
print("MANGO-LEVEL GROUP CROSS-VALIDATION")
print("=" * 65)

data = pd.read_csv(TRAIN_FILE)

TARGET = "spoilage_risk"
GROUP = "mango_id"

if TARGET not in data.columns or GROUP not in data.columns:
    raise ValueError("Required target or group column is missing.")

print("\nDATASET")
print("Observations:", len(data))
print("Unique mangoes:", data[GROUP].nunique())

# ------------------------------------------------------------
# 3. FEATURES
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
    f for f in FULL_FEATURES
    if f not in ["peel_color", "firmness"]
]

DAY_ONLY_FEATURES = ["storage_day"]

for column in FULL_FEATURES + [TARGET, GROUP]:
    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")

# ------------------------------------------------------------
# 4. ENCODE TARGET
# ------------------------------------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(data[TARGET].astype(str))
groups = data[GROUP].astype(str).to_numpy()

class_names = encoder.classes_
n_classes = len(class_names)

if n_classes != 5:
    raise ValueError(f"Expected 5 classes; found {n_classes}.")

print("\nCLASSES")
for i, name in enumerate(class_names):
    print(i, "=", name)

# ------------------------------------------------------------
# 5. PREPROCESSOR
# ------------------------------------------------------------

def make_preprocessor(features):

    numeric = [
        f for f in NUMERIC_FEATURES if f in features
    ]

    categorical = [
        f for f in CATEGORICAL_FEATURES if f in features
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
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

# ------------------------------------------------------------
# 6. MODEL
# ------------------------------------------------------------

def make_model(features):

    if features == DAY_ONLY_FEATURES:
        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist"
        )
    else:
        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist"
        )

    return Pipeline([
        ("preprocessor", make_preprocessor(features)),
        ("model", estimator)
    ])

# ------------------------------------------------------------
# 7. STRATIFIED GROUP FOLDS
# ------------------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model_definitions = {
    "FULL_FEATURE": FULL_FEATURES,
    "WITHOUT_FIRMNESS_AND_PEEL": REDUCED_FEATURES,
    "STORAGE_DAY_ONLY": DAY_ONLY_FEATURES
}

fold_results = []
prediction_results = []

# ------------------------------------------------------------
# 8. CROSS-VALIDATION
# ------------------------------------------------------------

for model_name, features in model_definitions.items():

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    X = data[features].copy()

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        if train_groups.intersection(val_groups):
            raise ValueError(
                f"Group leakage detected in fold {fold}."
            )

        if len(np.unique(y_train)) != n_classes:
            raise ValueError(
                f"Fold {fold} training data lacks a class. "
                "Review group distribution before continuing."
            )

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model = make_model(features)

        model.fit(
            X_train,
            y_train,
            model__sample_weight=weights
        )

        predictions = model.predict(X_val)

        metrics = {
            "model": model_name,
            "fold": fold,
            "training_mangoes": len(train_groups),
            "validation_mangoes": len(val_groups),
            "training_observations": len(train_idx),
            "validation_observations": len(val_idx),
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "macro_precision": precision_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ),
            "macro_recall": recall_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ),
            "macro_f1": f1_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ),
            "weighted_f1": f1_score(
                y_val, predictions,
                average="weighted",
                zero_division=0
            )
        }

        fold_results.append(metrics)

        prediction_results.append(pd.DataFrame({
            "mango_id": groups[val_idx],
            "actual_risk": encoder.inverse_transform(y_val),
            "predicted_risk": encoder.inverse_transform(
                predictions
            ),
            "correct": predictions == y_val,
            "model": model_name,
            "fold": fold
        }))

        print(f"\nFold {fold}")
        print("Training mangoes:", len(train_groups))
        print("Validation mangoes:", len(val_groups))
        print("Accuracy:", round(metrics["accuracy"], 4))
        print(
            "Balanced accuracy:",
            round(metrics["balanced_accuracy"], 4)
        )
        print("Macro F1:", round(metrics["macro_f1"], 4))

# ------------------------------------------------------------
# 9. SAVE FOLD RESULTS
# ------------------------------------------------------------

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    RESULTS / "step10E_fold_metrics.csv",
    index=False
)

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)

predictions_df.to_csv(
    RESULTS / "step10E_cv_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 10. SUMMARY: MEAN AND STANDARD DEVIATION
# ------------------------------------------------------------

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1"
]

summary_rows = []

for model_name, group_df in fold_df.groupby("model"):

    row = {"model": model_name}

    for metric in metric_columns:
        row[f"{metric}_mean"] = group_df[metric].mean()
        row[f"{metric}_std"] = group_df[metric].std(ddof=1)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULTS / "step10E_cv_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 11. REPORT
# ------------------------------------------------------------

report_path = RESULTS / "step10E_validation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("PHYSITWIN-X STEP 10E\n")
    file.write("MANGO-LEVEL GROUP CROSS-VALIDATION\n")
    file.write("=" * 65 + "\n\n")

    file.write("Training observations: " + str(len(data)) + "\n")
    file.write(
        "Training mangoes: " + str(data[GROUP].nunique()) + "\n"
    )
    file.write("Number of folds: 5\n")
    file.write("Grouping variable: mango_id\n\n")

    file.write("FOLD METRICS\n")
    file.write(fold_df.to_string(index=False))

    file.write("\n\nMEAN AND STANDARD DEVIATION\n")
    file.write(summary_df.to_string(index=False))

    file.write("\n\nINTERPRETATION NOTES\n")
    file.write(
        "Cross-validation was conducted using training mangoes only.\n"
    )
    file.write(
        "All observations from one mango were assigned to one fold.\n"
    )
    file.write(
        "The original 12-mango test set was not used in this step.\n"
    )
    file.write(
        "Mean and standard deviation describe performance across "
        "the five validation folds.\n"
    )
    file.write(
        "These results do not establish external generalization "
        "or independent future-spoilage prediction.\n"
    )

print("\n" + "=" * 65)
print("STEP 10E COMPLETED")
print("=" * 65)

print("\nCROSS-VALIDATION SUMMARY")
print(summary_df.round(4).to_string(index=False))

print("\nFILES SAVED:")
print("results/step10E_fold_metrics.csv")
print("results/step10E_cv_predictions.csv")
print("results/step10E_cv_summary.csv")
print("results/step10E_validation_report.txt")

print("\nSTEP 10E FINISHED.")
