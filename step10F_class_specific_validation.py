
# ============================================================
# PHYSITWIN-X STEP 10F
# CLASS-SPECIFIC VALIDATION ANALYSIS
# Corrected Version
# ============================================================

from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

INPUT_FILE = RESULTS / "step10E_cv_predictions.csv"

CLASSES = [
    "FRESH",
    "HIGH RISK",
    "LOW RISK",
    "MODERATE RISK",
    "VERY HIGH RISK"
]

MODELS = [
    "FULL_FEATURE",
    "WITHOUT_FIRMNESS_AND_PEEL",
    "STORAGE_DAY_ONLY"
]

EXPECTED_OBSERVATIONS = 602
EXPECTED_MANGOES = 48
EXPECTED_FOLDS = 5

RESULTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("PHYSITWIN-X STEP 10F")
print("CLASS-SPECIFIC VALIDATION ANALYSIS")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "mango_id",
    "actual_risk",
    "predicted_risk",
    "correct",
    "model",
    "fold"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nDATASET INFORMATION")
print("Total predictions:", len(data))
print("Models:", data["model"].nunique())
print("Folds:", data["fold"].nunique())
print("Unique mangoes:", data["mango_id"].nunique())

# ============================================================
# 3. GENERAL DATA VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("DATA VALIDATION")
print("=" * 65)

if set(data["model"].unique()) != set(MODELS):
    raise ValueError("Unexpected model names found.")

if set(data["actual_risk"].unique()) != set(CLASSES):
    raise ValueError("Unexpected actual risk classes found.")

if set(data["predicted_risk"].unique()) - set(CLASSES):
    raise ValueError("Unexpected predicted risk classes found.")

if data[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in required prediction columns."
    )

if sorted(data["fold"].unique()) != list(range(1, 6)):
    raise ValueError(
        "Expected fold numbers 1 through 5."
    )

# Check each model separately.
# Multiple daily observations per mango are expected.
for model_name in MODELS:

    subset = data[data["model"] == model_name]

    if len(subset) != EXPECTED_OBSERVATIONS:
        raise ValueError(
            f"{model_name}: expected "
            f"{EXPECTED_OBSERVATIONS} predictions, "
            f"found {len(subset)}."
        )

    if subset["mango_id"].nunique() != EXPECTED_MANGOES:
        raise ValueError(
            f"{model_name}: unexpected mango count."
        )

    if subset["fold"].nunique() != EXPECTED_FOLDS:
        raise ValueError(
            f"{model_name}: expected five folds."
        )

    # Verify that every mango belongs to only one validation fold.
    mango_fold_counts = (
        subset.groupby("mango_id")["fold"].nunique()
    )

    if (mango_fold_counts != 1).any():
        raise ValueError(
            f"{model_name}: a mango appears in multiple "
            "validation folds."
        )

    # Check the stored correctness column.
    expected_correct = (
        subset["actual_risk"].astype(str).str.strip()
        ==
        subset["predicted_risk"].astype(str).str.strip()
    )

    stored_correct = (
        subset["correct"].astype(str).str.strip().str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })
    )

    if stored_correct.isna().any():
        raise ValueError(
            f"{model_name}: invalid values in correct column."
        )

    if not (expected_correct.to_numpy() ==
            stored_correct.to_numpy()).all():
        raise ValueError(
            f"{model_name}: correctness column does not "
            "match actual and predicted labels."
        )

    print(f"{model_name}: PASSED")

print("\nGENERAL VALIDATION: PASSED")

# ============================================================
# 4. POOLED CLASS-SPECIFIC METRICS
# ============================================================

print("\n" + "=" * 65)
print("POOLED CLASS-SPECIFIC METRICS")
print("=" * 65)

pooled_results = []

for model_name in MODELS:

    subset = data[data["model"] == model_name]

    y_true = subset["actual_risk"]
    y_pred = subset["predicted_risk"]

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    for i, class_name in enumerate(CLASSES):

        pooled_results.append({
            "model": model_name,
            "risk_class": class_name,
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": int(support[i])
        })

pooled_df = pd.DataFrame(pooled_results)

pooled_file = (
    RESULTS / "step10F_pooled_class_metrics.csv"
)

pooled_df.to_csv(pooled_file, index=False)

print(pooled_df.round(4).to_string(index=False))

# ============================================================
# 5. CLASS-SPECIFIC METRICS PER FOLD
# ============================================================

print("\n" + "=" * 65)
print("FOLD-LEVEL CLASS METRICS")
print("=" * 65)

fold_results = []

for model_name in MODELS:

    for fold_number in range(1, EXPECTED_FOLDS + 1):

        subset = data[
            (data["model"] == model_name) &
            (data["fold"] == fold_number)
        ]

        y_true = subset["actual_risk"]
        y_pred = subset["predicted_risk"]

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=CLASSES,
                zero_division=0
            )
        )

        for i, class_name in enumerate(CLASSES):

            fold_results.append({
                "model": model_name,
                "fold": fold_number,
                "risk_class": class_name,
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": int(support[i])
            })

fold_df = pd.DataFrame(fold_results)

fold_file = (
    RESULTS / "step10F_fold_class_metrics.csv"
)

fold_df.to_csv(fold_file, index=False)

print(fold_df.round(4).to_string(index=False))

# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRICES")
print("=" * 65)

for model_name in MODELS:

    subset = data[data["model"] == model_name]

    matrix = confusion_matrix(
        subset["actual_risk"],
        subset["predicted_risk"],
        labels=CLASSES
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES
    )

    output_file = (
        RESULTS /
        f"step10F_{model_name.lower()}_confusion_matrix.csv"
    )

    matrix_df.to_csv(output_file)

    print(f"\n{model_name}")
    print(matrix_df)

# ============================================================
# 7. CROSS-VALIDATION CLASS SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CROSS-VALIDATION CLASS SUMMARY")
print("=" * 65)

fold_summary = (
    fold_df
    .groupby(["model", "risk_class"])
    .agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1_score", "mean"),
        f1_std=("f1_score", "std"),
        total_support=("support", "sum")
    )
    .reset_index()
)

summary_file = (
    RESULTS / "step10F_class_fold_summary.csv"
)

fold_summary.to_csv(summary_file, index=False)

print(fold_summary.round(4).to_string(index=False))

# ============================================================
# 8. WRITE VALIDATION REPORT
# ============================================================

report_file = (
    RESULTS / "step10F_class_validation_report.txt"
)

with open(report_file, "w", encoding="utf-8") as file:

    file.write("PHYSITWIN-X STEP 10F\n")
    file.write("CLASS-SPECIFIC VALIDATION ANALYSIS\n")
    file.write("=" * 65 + "\n\n")

    file.write("DATASET INFORMATION\n")
    file.write(f"Total predictions: {len(data)}\n")
    file.write(f"Unique mangoes: {data['mango_id'].nunique()}\n")
    file.write(f"Number of models: {data['model'].nunique()}\n")
    file.write(f"Number of folds: {data['fold'].nunique()}\n\n")

    file.write("POOLED CLASS-SPECIFIC METRICS\n")
    file.write(pooled_df.to_string(index=False))

    file.write("\n\nFOLD-LEVEL CLASS METRICS\n")
    file.write(fold_df.to_string(index=False))

    file.write("\n\nCROSS-VALIDATION CLASS SUMMARY\n")
    file.write(fold_summary.to_string(index=False))

    file.write("\n\nINTERPRETATION NOTES\n")
    file.write(
        "Pooled metrics combine out-of-fold predictions "
        "from the five validation folds.\n"
    )
    file.write(
        "Fold-level metrics describe class-specific performance "
        "within each validation fold.\n"
    )
    file.write(
        "A class may have zero support in a particular fold. "
        "Its metrics for that fold should be interpreted cautiously.\n"
    )
    file.write(
        "These metrics assess reproduction of recorded spoilage-risk "
        "labels, not independent prediction of future spoilage.\n"
    )
    file.write(
        "Firmness and peel color may be related to how the original "
        "risk labels were assigned. This dependency must be considered "
        "when interpreting classification performance.\n"
    )

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STEP 10F COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFILES SAVED:")

files = [
    "step10F_pooled_class_metrics.csv",
    "step10F_fold_class_metrics.csv",
    "step10F_class_fold_summary.csv",
    "step10F_full_feature_confusion_matrix.csv",
    "step10F_without_firmness_and_peel_confusion_matrix.csv",
    "step10F_storage_day_only_confusion_matrix.csv",
    "step10F_class_validation_report.txt"
]

for filename in files:
    print(f"results/{filename}")

print("\nALL STEP 10F ANALYSES FINISHED.")
print("=" * 65)
