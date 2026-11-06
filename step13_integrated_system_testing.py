
# ================================================================
# PHYSITWIN-X
# STEP 13: INTEGRATED SYSTEM TESTING
# ================================================================
# Purpose:
# 1. Verify Step 12 integration files.
# 2. Validate dataset and prediction integrity.
# 3. Recompute predictions using saved models.
# 4. Verify risk probabilities.
# 5. Verify latest mango status.
# 6. Verify daily summary.
#
# IMPORTANT:
# - Does NOT retrain models.
# - Does NOT overwrite trained models.
# - Does NOT modify original datasets.
# ================================================================

from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
import joblib


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

STEP12_DIR = RESULTS_DIR / "step12_digital_twin"
STEP13_DIR = RESULTS_DIR / "step13_system_testing"

STEP13_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = RESULTS_DIR / "figures"

DYNAMIC_DATA = DATA_DIR / "physitwin_x_dynamic_states.csv"

SHELF_MODEL_PATH = (
    RESULTS_DIR / "physitwin_x_xgboost_model.joblib"
)

RISK_MODEL_PATH = (
    RESULTS_DIR / "physitwin_x_spoilage_risk_model.joblib"
)

INTEGRATED_FILE = (
    STEP12_DIR / "physitwin_x_integrated_predictions.csv"
)

LATEST_FILE = (
    STEP12_DIR / "physitwin_x_latest_mango_status.csv"
)

RISK_DISTRIBUTION_FILE = (
    STEP12_DIR / "step12_risk_distribution.csv"
)

DAILY_SUMMARY_FILE = (
    STEP12_DIR / "step12_daily_digital_twin_summary.csv"
)

REPORT_FILE = (
    STEP13_DIR / "step13_system_testing_report.txt"
)

CHECKS_FILE = (
    STEP13_DIR / "step13_validation_checks.csv"
)


# ================================================================
# 2. EXPECTED MODEL FEATURES
# ================================================================

SHELF_FEATURES = [
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
    "humidity_change_pct",
]

RISK_FEATURES = [
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
    "firmness",
]

RISK_PROBABILITY_COLUMNS = {
    "FRESH": "probability_fresh",
    "LOW RISK": "probability_low_risk",
    "MODERATE RISK": "probability_moderate_risk",
    "HIGH RISK": "probability_high_risk",
    "VERY HIGH RISK": "probability_very_high_risk",
}


# ================================================================
# 3. VALIDATION TRACKER
# ================================================================

checks = []
errors = []


def record_check(name, passed, details=""):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details)
    })

    symbol = "PASS" if passed else "FAIL"

    print(f"[{symbol}] {name}")

    if details:
        print(f"       {details}")

    if not passed:
        errors.append(name)


def require_columns(dataframe, columns, name):
    missing = [
        column for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )


def normalize_risk(value):
    return str(value).strip().upper().replace("_", " ")


# ================================================================
# 4. START TESTING
# ================================================================

print("\n" + "=" * 70)
print("PHYSITWIN-X: STEP 13 INTEGRATED SYSTEM TESTING")
print("=" * 70)

print(f"Project directory: {BASE_DIR}")
print(f"Testing started: {datetime.now()}")


try:

    # ------------------------------------------------------------
    # TEST 1: REQUIRED FILES
    # ------------------------------------------------------------

    print("\nTEST 1: REQUIRED FILE VERIFICATION")

    required_files = {
        "Dynamic state dataset": DYNAMIC_DATA,
        "Shelf-life model": SHELF_MODEL_PATH,
        "Spoilage-risk model": RISK_MODEL_PATH,
        "Integrated predictions": INTEGRATED_FILE,
        "Latest mango status": LATEST_FILE,
        "Risk distribution": RISK_DISTRIBUTION_FILE,
        "Daily summary": DAILY_SUMMARY_FILE,
    }

    for name, path in required_files.items():
        record_check(
            f"File exists: {name}",
            path.exists(),
            path
        )

    missing_files = [
        name for name, path in required_files.items()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files: {missing_files}"
        )

    # ------------------------------------------------------------
    # TEST 2: LOAD DATA AND MODELS
    # ------------------------------------------------------------

    print("\nTEST 2: LOAD DATA AND SAVED MODELS")

    source = pd.read_csv(DYNAMIC_DATA)

    integrated = pd.read_csv(INTEGRATED_FILE)

    latest = pd.read_csv(LATEST_FILE)

    risk_distribution = pd.read_csv(
        RISK_DISTRIBUTION_FILE
    )

    daily_summary = pd.read_csv(
        DAILY_SUMMARY_FILE
    )

    shelf_model = joblib.load(SHELF_MODEL_PATH)

    risk_package = joblib.load(RISK_MODEL_PATH)

    risk_model = risk_package["model"]

    label_encoder = risk_package["label_encoder"]

    risk_model_features = risk_package["feature_columns"]

    record_check(
        "Source dataset loaded",
        len(source) > 0,
        f"{len(source)} observations"
    )

    record_check(
        "Integrated predictions loaded",
        len(integrated) > 0,
        f"{len(integrated)} observations"
    )

    record_check(
        "Saved shelf-life model loaded",
        shelf_model is not None
    )

    record_check(
        "Saved risk model loaded",
        risk_model is not None
    )

    # ------------------------------------------------------------
    # TEST 3: DATASET SIZE AND MANGO COUNT
    # ------------------------------------------------------------

    print("\nTEST 3: DATASET INTEGRITY")

    require_columns(
        source,
        ["mango_id", "storage_day"],
        "Source dataset"
    )

    require_columns(
        integrated,
        [
            "mango_id",
            "storage_day",
            "predicted_remaining_shelf_life_days",
            "predicted_spoilage_risk",
            "maximum_risk_class_probability",
        ],
        "Integrated predictions"
    )

    record_check(
        "Expected 745 source observations",
        len(source) == 745,
        f"Actual: {len(source)}"
    )

    record_check(
        "Expected 745 integrated observations",
        len(integrated) == 745,
        f"Actual: {len(integrated)}"
    )

    record_check(
        "Expected 60 unique mangoes",
        source["mango_id"].nunique() == 60,
        f"Actual: {source['mango_id'].nunique()}"
    )

    record_check(
        "Integrated output contains 60 mangoes",
        integrated["mango_id"].nunique() == 60,
        f"Actual: {integrated['mango_id'].nunique()}"
    )

    record_check(
        "Storage days cover Day 1 to Day 14",
        source["storage_day"].min() == 1
        and source["storage_day"].max() == 14,
        f"Range: {source['storage_day'].min()} "
        f"to {source['storage_day'].max()}"
    )

    source_duplicates = source.duplicated(
        subset=["mango_id", "storage_day"]
    ).sum()

    integrated_duplicates = integrated.duplicated(
        subset=["mango_id", "storage_day"]
    ).sum()

    record_check(
        "No duplicate source mango/day records",
        source_duplicates == 0,
        f"Duplicates: {source_duplicates}"
    )

    record_check(
        "No duplicate integrated mango/day records",
        integrated_duplicates == 0,
        f"Duplicates: {integrated_duplicates}"
    )

    # ------------------------------------------------------------
    # TEST 4: SOURCE AND INTEGRATED RECORD MATCHING
    # ------------------------------------------------------------

    print("\nTEST 4: SOURCE-INTEGRATION RECORD MATCHING")

    source_keys = set(
        zip(source["mango_id"], source["storage_day"])
    )

    integrated_keys = set(
        zip(integrated["mango_id"], integrated["storage_day"])
    )

    record_check(
        "All source records retained in integration",
        source_keys == integrated_keys,
        f"Source keys: {len(source_keys)}; "
        f"Integrated keys: {len(integrated_keys)}"
    )

    # ------------------------------------------------------------
    # TEST 5: SHELF-LIFE PREDICTION VALIDATION
    # ------------------------------------------------------------

    print("\nTEST 5: SHELF-LIFE PREDICTION VALIDATION")

    shelf_predictions = pd.to_numeric(
        integrated["predicted_remaining_shelf_life_days"],
        errors="coerce"
    )

    record_check(
        "Shelf-life predictions contain no missing values",
        shelf_predictions.notna().all()
    )

    record_check(
        "Shelf-life predictions are finite",
        np.isfinite(shelf_predictions).all()
    )

    negative_count = int((shelf_predictions < 0).sum())

    record_check(
        "Negative predictions identified and preserved",
        True,
        f"{negative_count} raw predictions below zero"
    )

    # ------------------------------------------------------------
    # TEST 6: RECOMPUTE SHELF-LIFE PREDICTIONS
    # ------------------------------------------------------------

    print("\nTEST 6: SHELF-LIFE MODEL REPRODUCIBILITY")

    require_columns(
        integrated,
        SHELF_FEATURES,
        "Integrated shelf-life input"
    )

    shelf_input = integrated[SHELF_FEATURES].copy()

    recomputed_shelf = shelf_model.predict(
        shelf_input
    )

    shelf_difference = np.abs(
        recomputed_shelf - shelf_predictions.to_numpy()
    )

    max_shelf_difference = float(
        np.max(shelf_difference)
    )

    shelf_match = np.allclose(
        recomputed_shelf,
        shelf_predictions.to_numpy(),
        rtol=1e-5,
        atol=1e-5
    )

    record_check(
        "Shelf-life predictions reproducible",
        shelf_match,
        f"Maximum absolute difference: "
        f"{max_shelf_difference:.10f}"
    )

    # ------------------------------------------------------------
    # TEST 7: RISK CLASSIFICATION VALIDATION
    # ------------------------------------------------------------

    print("\nTEST 7: SPOILAGE-RISK CLASSIFICATION")

    expected_classes = {
        "FRESH",
        "LOW RISK",
        "MODERATE RISK",
        "HIGH RISK",
        "VERY HIGH RISK",
    }

    predicted_classes = integrated[
        "predicted_spoilage_risk"
    ].map(normalize_risk)

    actual_classes = set(predicted_classes.unique())

    record_check(
        "All predicted risk classes are recognized",
        actual_classes.issubset(expected_classes),
        f"Observed classes: {sorted(actual_classes)}"
    )

    record_check(
        "No missing risk predictions",
        predicted_classes.notna().all()
    )

    # ------------------------------------------------------------
    # TEST 8: RISK PROBABILITY VALIDATION
    # ------------------------------------------------------------

    print("\nTEST 8: RISK PROBABILITY VALIDATION")

    probability_columns = list(
        RISK_PROBABILITY_COLUMNS.values()
    )

    require_columns(
        integrated,
        probability_columns,
        "Integrated risk probabilities"
    )

    probabilities = integrated[
        probability_columns
    ].apply(pd.to_numeric, errors="coerce")

    record_check(
        "No missing risk probabilities",
        probabilities.notna().all().all()
    )

    probability_values = probabilities.to_numpy()

    record_check(
        "All risk probabilities are finite",
        np.isfinite(probability_values).all()
    )

    record_check(
        "Risk probabilities are between zero and one",
        (
            (probability_values >= -1e-8)
            & (probability_values <= 1 + 1e-8)
        ).all()
    )

    probability_sums = probabilities.sum(axis=1)

    record_check(
        "Risk probabilities sum approximately to one",
        np.allclose(
            probability_sums,
            1.0,
            atol=1e-5
        ),
        f"Maximum sum deviation: "
        f"{np.max(np.abs(probability_sums - 1)):.10f}"
    )

    maximum_probability = probabilities.max(axis=1)

    recorded_maximum = pd.to_numeric(
        integrated["maximum_risk_class_probability"],
        errors="coerce"
    )

    record_check(
        "Maximum class probability is correctly recorded",
        np.allclose(
            maximum_probability.to_numpy(),
            recorded_maximum.to_numpy(),
            rtol=1e-5,
            atol=1e-5
        )
    )

    # ------------------------------------------------------------
    # TEST 9: RECOMPUTE RISK MODEL PREDICTIONS
    # ------------------------------------------------------------

    print("\nTEST 9: RISK MODEL REPRODUCIBILITY")

    require_columns(
        integrated,
        RISK_FEATURES,
        "Integrated risk-model input"
    )

    if list(risk_model_features) != RISK_FEATURES:
        raise ValueError(
            "Saved risk model feature order does not match "
            "the verified Step 12 feature order."
        )

    risk_input = integrated[RISK_FEATURES].copy()

    recomputed_probabilities = risk_model.predict_proba(
        risk_input
    )

    recomputed_class_indices = np.argmax(
        recomputed_probabilities,
        axis=1
    )

    recomputed_risk = label_encoder.inverse_transform(
        recomputed_class_indices
    )

    recomputed_risk = pd.Series(
        recomputed_risk
    ).map(normalize_risk)

    risk_match = np.array_equal(
        recomputed_risk.to_numpy(),
        predicted_classes.to_numpy()
    )

    record_check(
        "Risk classifications reproducible",
        risk_match
    )

    probability_match = True

    for class_index, encoded_class in enumerate(
        label_encoder.classes_
    ):
        class_name = normalize_risk(encoded_class)

        probability_column = RISK_PROBABILITY_COLUMNS[
            class_name
        ]

        stored = pd.to_numeric(
            integrated[probability_column],
            errors="coerce"
        ).to_numpy()

        recalculated = recomputed_probabilities[
            :, class_index
        ]

        if not np.allclose(
            stored,
            recalculated,
            rtol=1e-5,
            atol=1e-5
        ):
            probability_match = False

    record_check(
        "Risk probabilities reproducible",
        probability_match
    )

    # ------------------------------------------------------------
    # TEST 10: LATEST MANGO STATUS
    # ------------------------------------------------------------

    print("\nTEST 10: LATEST MANGO STATUS")

    require_columns(
        latest,
        [
            "mango_id",
            "storage_day",
            "predicted_remaining_shelf_life_days",
            "predicted_spoilage_risk",
        ],
        "Latest mango status"
    )

    latest_duplicates = latest["mango_id"].duplicated().sum()

    record_check(
        "Latest status has one record per mango",
        latest_duplicates == 0
        and len(latest) == 60,
        f"Rows: {len(latest)}; "
        f"duplicate mango IDs: {latest_duplicates}"
    )

    expected_latest = (
        integrated
        .sort_values(["mango_id", "storage_day"])
        .groupby("mango_id", as_index=False)
        .tail(1)
    )

    expected_latest = expected_latest.set_index(
        "mango_id"
    ).sort_index()

    actual_latest = latest.set_index(
        "mango_id"
    ).sort_index()

    latest_ids_match = (
        set(expected_latest.index)
        == set(actual_latest.index)
    )

    record_check(
        "Latest status contains the correct mango IDs",
        latest_ids_match
    )

    latest_days_match = False

    if latest_ids_match:
        latest_days_match = np.array_equal(
            expected_latest["storage_day"].to_numpy(),
            actual_latest["storage_day"].to_numpy()
        )

    record_check(
        "Latest status matches final recorded day",
        latest_days_match
    )

    latest_shelf_match = False
    latest_risk_match = False

    if latest_ids_match:
        latest_shelf_match = np.allclose(
            expected_latest[
                "predicted_remaining_shelf_life_days"
            ].to_numpy(),
            actual_latest[
                "predicted_remaining_shelf_life_days"
            ].to_numpy(),
            rtol=1e-5,
            atol=1e-5
        )

        latest_risk_match = (
            expected_latest[
                "predicted_spoilage_risk"
            ].map(normalize_risk).to_numpy()
            == actual_latest[
                "predicted_spoilage_risk"
            ].map(normalize_risk).to_numpy()
        ).all()

    record_check(
        "Latest shelf-life predictions match integration",
        latest_shelf_match
    )

    record_check(
        "Latest risk classifications match integration",
        latest_risk_match
    )

    # ------------------------------------------------------------
    # TEST 11: DAILY SUMMARY
    # ------------------------------------------------------------

    print("\nTEST 11: DAILY SUMMARY VALIDATION")

    record_check(
        "Daily summary is not empty",
        len(daily_summary) > 0,
        f"Rows: {len(daily_summary)}"
    )

    if "storage_day" in daily_summary.columns:
        record_check(
            "Daily summary has unique storage days",
            daily_summary["storage_day"].is_unique
        )

        record_check(
            "Daily summary covers the recorded period",
            daily_summary["storage_day"].min() == 1
            and daily_summary["storage_day"].max() == 14
        )
    else:
        record_check(
            "Daily summary contains storage_day",
            False
        )

    record_check(
        "Risk distribution file is available",
        len(risk_distribution) > 0
    )

    # ------------------------------------------------------------
    # TEST 12: DESCRIPTIVE LABEL AGREEMENT
    # ------------------------------------------------------------

    print("\nTEST 12: RECORDED LABEL AGREEMENT")

    if "spoilage_risk" in integrated.columns:

        recorded_risk = integrated[
            "spoilage_risk"
        ].map(normalize_risk)

        agreement = (
            recorded_risk.to_numpy()
            == predicted_classes.to_numpy()
        )

        agreement_rate = float(
            agreement.mean() * 100
        )

        record_check(
            "Recorded risk-label agreement calculated",
            True,
            f"{agreement.sum()}/{len(agreement)} "
            f"({agreement_rate:.2f}%)"
        )

        print(
            "NOTE: This is descriptive agreement with "
            "recorded labels, NOT independent validation."
        )

    else:
        agreement_rate = None

        record_check(
            "Recorded risk-label agreement",
            True,
            "Skipped: original risk label not present."
        )

    # ------------------------------------------------------------
    # TEST 13: SAVE VALIDATION RESULTS
    # ------------------------------------------------------------

    print("\nTEST 13: SAVE TESTING RESULTS")

    checks_df = pd.DataFrame(checks)

    checks_df.to_csv(
        CHECKS_FILE,
        index=False
    )

    passed_count = sum(
        item["status"] == "PASS"
        for item in checks
    )

    failed_count = sum(
        item["status"] == "FAIL"
        for item in checks
    )

    total_checks = len(checks)

    overall_status = (
        "PASSED"
        if failed_count == 0
        else "FAILED"
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as report:

        report.write("=" * 70 + "\n")
        report.write("PHYSITWIN-X STEP 13 SYSTEM TESTING REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write(
            f"Testing date: {datetime.now()}\n"
        )

        report.write(
            f"Overall status: {overall_status}\n\n"
        )

        report.write("DATASET INFORMATION\n")
        report.write("-" * 50 + "\n")

        report.write(
            f"Observations: {len(source)}\n"
        )

        report.write(
            f"Unique mangoes: {source['mango_id'].nunique()}\n"
        )

        report.write(
            f"Storage period: "
            f"{source['storage_day'].min()} to "
            f"{source['storage_day'].max()}\n\n"
        )

        report.write("SHELF-LIFE PREDICTION\n")
        report.write("-" * 50 + "\n")

        report.write(
            f"Mean prediction: "
            f"{shelf_predictions.mean():.4f} days\n"
        )

        report.write(
            f"Minimum raw prediction: "
            f"{shelf_predictions.min():.4f} days\n"
        )

        report.write(
            f"Maximum prediction: "
            f"{shelf_predictions.max():.4f} days\n"
        )

        report.write(
            f"Negative raw predictions: {negative_count}\n\n"
        )

        report.write("RISK CLASSIFICATION\n")
        report.write("-" * 50 + "\n")

        for class_name, count in (
            predicted_classes.value_counts().items()
        ):
            report.write(
                f"{class_name}: {count}\n"
            )

        report.write("\n")

        report.write("REPRODUCIBILITY\n")
        report.write("-" * 50 + "\n")

        report.write(
            f"Shelf-life maximum difference: "
            f"{max_shelf_difference:.10f}\n"
        )

        report.write(
            f"Risk class reproduction: {risk_match}\n"
        )

        report.write(
            f"Risk probability reproduction: "
            f"{probability_match}\n\n"
        )

        report.write("VALIDATION SUMMARY\n")
        report.write("-" * 50 + "\n")

        report.write(
            f"Total checks: {total_checks}\n"
        )

        report.write(
            f"Passed: {passed_count}\n"
        )

        report.write(
            f"Failed: {failed_count}\n"
        )

        report.write(
            f"Overall status: {overall_status}\n\n"
        )

        report.write("INDIVIDUAL CHECKS\n")
        report.write("-" * 50 + "\n")

        for item in checks:
            report.write(
                f"[{item['status']}] {item['check']}\n"
            )

            if item["details"]:
                report.write(
                    f"    {item['details']}\n"
                )

        report.write("\n")
        report.write(
            "SCIENTIFIC LIMITATIONS\n"
        )

        report.write("-" * 50 + "\n")

        report.write(
            "1. Reproducibility confirms that saved models "
            "produce the same stored predictions.\n"
        )

        report.write(
            "2. Reproducibility does not establish external "
            "generalization or independent predictive accuracy.\n"
        )

        report.write(
            "3. Recorded risk-label agreement is descriptive "
            "and is not independent validation.\n"
        )

        report.write(
            "4. Negative raw shelf-life predictions are retained "
            "for research transparency.\n"
        )

        report.write(
            "5. Maximum predicted class probability is not "
            "the probability that a mango is spoiled.\n"
        )

        report.write(
            "6. This test does not establish live sensor "
            "connectivity or automatic real-time updating.\n"
        )

    print("\n" + "=" * 70)
    print("STEP 13 TESTING COMPLETED")
    print("=" * 70)

    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    print(f"\nOVERALL STATUS: {overall_status}")

    print("\nOUTPUT FILES:")
    print(CHECKS_FILE)
    print(REPORT_FILE)

    if failed_count:
        print("\nFAILED CHECKS:")

        for name in errors:
            print("-", name)

    else:
        print("\nALL SYSTEM TESTS PASSED.")

except Exception as error:

    print("\n" + "=" * 70)
    print("STEP 13 TESTING ERROR")
    print("=" * 70)

    print(str(error))
    traceback.print_exc()

    # Save checks completed before the error.
    if checks:
        pd.DataFrame(checks).to_csv(
            CHECKS_FILE,
            index=False
        )

    raise
