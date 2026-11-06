
# ============================================================
# PHYSITWIN-X STEP 11
# SHAP EXPLAINABILITY FOR SPOILAGE-RISK CLASSIFICATION
# CORRECTED MODEL-COMPATIBLE VERSION
#
# Uses the existing Step 10C trained model.
# Does NOT retrain or overwrite the model.
# ============================================================

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_FILE = (
    RESULTS_DIR / "physitwin_x_spoilage_risk_model.joblib"
)

TEST_FILE = (
    DATA_DIR / "physitwin_x_risk_test.csv"
)

OUTPUT_DIR = RESULTS_DIR / "step11_shap"
OUTPUT_FIGURES = FIGURES_DIR / "step11_shap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

EXPECTED_CLASSES = {
    "FRESH",
    "LOW RISK",
    "MODERATE RISK",
    "HIGH RISK",
    "VERY HIGH RISK"
}

# ============================================================
# 2. LOAD SAVED MODEL
# ============================================================

print("=" * 70)
print("PHYSITWIN-X STEP 11")
print("SHAP EXPLAINABILITY - SPOILAGE-RISK CLASSIFICATION")
print("=" * 70)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}"
    )

print("\nLoading Step 10C model...")

saved_object = joblib.load(MODEL_FILE)

if not isinstance(saved_object, dict):
    raise TypeError("Saved model must be a dictionary.")

required_keys = [
    "model",
    "label_encoder",
    "feature_columns",
    "classes"
]

for key in required_keys:
    if key not in saved_object:
        raise KeyError(
            f"Missing saved model component: {key}"
        )

model = saved_object["model"]
label_encoder = saved_object["label_encoder"]

# IMPORTANT:
# Use the exact feature order stored in Step 10C.
FEATURE_COLUMNS = list(
    saved_object["feature_columns"]
)

saved_classes = [
    str(x).strip().upper()
    for x in saved_object["classes"]
]

print("Model loaded successfully.")

print("\nSAVED FEATURE ORDER:")

for i, feature in enumerate(FEATURE_COLUMNS, start=1):
    print(f"{i}. {feature}")

print("\nSAVED CLASS ORDER:")
print(saved_classes)

if set(saved_classes) != EXPECTED_CLASSES:
    raise ValueError(
        "Saved model classes do not match the expected "
        "five spoilage-risk classes."
    )

# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\nLoading grouped test dataset...")

test_data = pd.read_csv(TEST_FILE)

TARGET_COLUMN = "spoilage_risk"

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column for column in required_columns
    if column not in test_data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing test dataset columns: {missing_columns}"
    )

X_test = test_data[FEATURE_COLUMNS].copy()

y_test = (
    test_data[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.upper()
)

if X_test.empty:
    raise ValueError("Test dataset is empty.")

if y_test.isna().any():
    raise ValueError("Missing target labels detected.")

print("\nTEST DATASET")
print("Observations:", len(X_test))

if "mango_id" in test_data.columns:
    print("Unique mangoes:", test_data["mango_id"].nunique())

print("\nACTUAL RISK DISTRIBUTION")
print(y_test.value_counts().to_string())

# ============================================================
# 4. VERIFY PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PIPELINE INSPECTION")
print("=" * 70)

if not isinstance(model, Pipeline):
    raise TypeError(
        "Expected a scikit-learn Pipeline."
    )

print("Pipeline steps:")

for name, component in model.steps:
    print(name, ":", type(component))

if len(model.steps) != 2:
    raise ValueError(
        "Unexpected pipeline structure. Expected preprocessor "
        "and classifier."
    )

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# ============================================================
# 5. TRANSFORM TEST DATA
# ============================================================

print("\nTransforming test data...")

X_transformed = preprocessor.transform(X_test)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(
    X_transformed,
    dtype=np.float64
)

try:
    feature_names = list(
        preprocessor.get_feature_names_out()
    )
except Exception as error:
    raise RuntimeError(
        "Unable to retrieve transformed feature names."
    ) from error

if X_transformed.shape[1] != len(feature_names):
    raise ValueError(
        "Transformed feature dimensions do not match feature names."
    )

print("Original predictors:", len(FEATURE_COLUMNS))
print("Transformed predictors:", X_transformed.shape[1])
print("Test observations:", X_transformed.shape[0])

# Save feature mapping for documentation.
feature_mapping = pd.DataFrame({
    "transformed_feature_index": range(len(feature_names)),
    "transformed_feature": feature_names
})

feature_mapping.to_csv(
    OUTPUT_DIR / "step11_transformed_feature_mapping.csv",
    index=False
)

# ============================================================
# 6. VERIFY CLASS ENCODING
# ============================================================

print("\n" + "=" * 70)
print("CLASS ENCODING")
print("=" * 70)

encoder_classes = [
    str(x).strip().upper()
    for x in label_encoder.classes_
]

print("Label encoder classes:")

for i, class_name in enumerate(encoder_classes):
    print(f"{i} = {class_name}")

classifier_classes = np.asarray(
    classifier.classes_,
    dtype=int
)

class_names = [
    str(x).strip().upper()
    for x in label_encoder.inverse_transform(
        classifier_classes
    )
]

if set(class_names) != EXPECTED_CLASSES:
    raise ValueError(
        "Classifier output classes do not match expected classes."
    )

print("\nClassifier probability order:")
print(class_names)

# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING CLASSIFICATION PREDICTIONS")
print("=" * 70)

predicted_encoded = model.predict(X_test)

predicted_labels = label_encoder.inverse_transform(
    np.asarray(predicted_encoded, dtype=int)
)

predicted_labels = [
    str(x).strip().upper()
    for x in predicted_labels
]

probabilities = model.predict_proba(X_test)

prediction_df = pd.DataFrame({
    "actual_risk": y_test.to_numpy(),
    "predicted_risk": predicted_labels,
    "correct": (
        y_test.to_numpy() == np.asarray(predicted_labels)
    )
})

if "mango_id" in test_data.columns:
    prediction_df.insert(
        0,
        "mango_id",
        test_data["mango_id"].to_numpy()
    )

if "storage_day" in test_data.columns:
    prediction_df.insert(
        1 if "mango_id" in test_data.columns else 0,
        "storage_day",
        test_data["storage_day"].to_numpy()
    )

for i, class_name in enumerate(class_names):

    safe_name = class_name.lower().replace(" ", "_")

    prediction_df[f"probability_{safe_name}"] = (
        probabilities[:, i]
    )

prediction_df.to_csv(
    OUTPUT_DIR / "step11_test_predictions.csv",
    index=False
)

print("Predictions saved.")

# ============================================================
# 8. CREATE SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

explainer = shap.TreeExplainer(classifier)

shap_explanation = explainer(
    X_transformed
)

shap_values = shap_explanation.values

if isinstance(shap_values, list):
    shap_array = np.stack(
        shap_values,
        axis=-1
    )
else:
    shap_array = np.asarray(shap_values)

if shap_array.ndim == 2:
    shap_array = shap_array[:, :, np.newaxis]

if shap_array.ndim != 3:
    raise ValueError(
        f"Unexpected SHAP dimensions: {shap_array.shape}"
    )

if shap_array.shape != (
    len(X_test),
    len(feature_names),
    len(class_names)
):
    raise ValueError(
        "SHAP dimensions do not match test observations, "
        "features, and classes. Actual dimensions: "
        f"{shap_array.shape}"
    )

print("SHAP values calculated successfully.")
print("SHAP dimensions:", shap_array.shape)

# ============================================================
# 9. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

global_importance = np.mean(
    np.abs(shap_array),
    axis=(0, 2)
)

global_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": global_importance
})

global_df = global_df.sort_values(
    "mean_absolute_shap",
    ascending=False
).reset_index(drop=True)

global_df.insert(
    0,
    "rank",
    np.arange(1, len(global_df) + 1)
)

global_df.to_csv(
    OUTPUT_DIR / "step11_shap_global_importance.csv",
    index=False
)

print(global_df.head(20).round(6).to_string(index=False))

# ============================================================
# 10. CLASS-SPECIFIC SHAP IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS-SPECIFIC SHAP IMPORTANCE")
print("=" * 70)

class_importance_rows = []

for class_index, class_name in enumerate(class_names):

    values = shap_array[:, :, class_index]

    importance = np.mean(
        np.abs(values),
        axis=0
    )

    for feature_index, feature_name in enumerate(feature_names):

        class_importance_rows.append({
            "risk_class": class_name,
            "feature": feature_name,
            "mean_absolute_shap": importance[feature_index]
        })

class_importance_df = pd.DataFrame(
    class_importance_rows
)

class_importance_df = class_importance_df.sort_values(
    ["risk_class", "mean_absolute_shap"],
    ascending=[True, False]
)

class_importance_df.to_csv(
    OUTPUT_DIR / "step11_shap_class_importance.csv",
    index=False
)

for class_name in class_names:

    print(f"\n{class_name}")

    print(
        class_importance_df[
            class_importance_df["risk_class"] == class_name
        ].head(10).round(6).to_string(index=False)
    )

# ============================================================
# 11. SAVE COMPLETE SHAP VALUES
# ============================================================

print("\nSaving complete SHAP values...")

shap_rows = []

for sample_index in range(len(X_test)):

    for feature_index, feature_name in enumerate(feature_names):

        row = {
            "sample_index": sample_index,
            "feature": feature_name,
            "transformed_value": X_transformed[
                sample_index, feature_index
            ]
        }

        for class_index, class_name in enumerate(class_names):

            safe_name = class_name.lower().replace(" ", "_")

            row[f"shap_{safe_name}"] = shap_array[
                sample_index,
                feature_index,
                class_index
            ]

        shap_rows.append(row)

shap_values_df = pd.DataFrame(shap_rows)

shap_values_df.to_csv(
    OUTPUT_DIR / "step11_shap_values.csv",
    index=False
)

# ============================================================
# 12. GLOBAL SHAP BAR PLOT
# ============================================================

print("\nGenerating global SHAP plot...")

top_global = global_df.head(20).sort_values(
    "mean_absolute_shap",
    ascending=True
)

plt.figure(figsize=(12, 9))

plt.barh(
    top_global["feature"],
    top_global["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Transformed Feature")

plt.title(
    "PhysiTwin-X Global SHAP Feature Importance\n"
    "Spoilage-Risk Classification"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURES / "step11_global_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 13. CLASS-SPECIFIC SHAP BAR PLOTS
# ============================================================

print("\nGenerating class-specific SHAP bar plots...")

for class_index, class_name in enumerate(class_names):

    importance = np.mean(
        np.abs(shap_array[:, :, class_index]),
        axis=0
    )

    class_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    class_df = class_df.nlargest(
        15,
        "importance"
    ).sort_values("importance")

    plt.figure(figsize=(11, 8))

    plt.barh(
        class_df["feature"],
        class_df["importance"]
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Transformed Feature")

    plt.title(
        f"SHAP Feature Importance: {class_name}"
    )

    plt.tight_layout()

    safe_name = class_name.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_FIGURES / f"step11_shap_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 14. CLASS-SPECIFIC SHAP SUMMARY PLOTS
# ============================================================

print("\nGenerating SHAP summary plots...")

for class_index, class_name in enumerate(class_names):

    class_values = shap_array[:, :, class_index]

    safe_name = class_name.lower().replace(" ", "_")

    plt.figure(figsize=(12, 9))

    shap.summary_plot(
        class_values,
        X_transformed,
        feature_names=feature_names,
        max_display=20,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot: {class_name}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES /
        f"step11_summary_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 15. INDIVIDUAL MANGO EXPLANATIONS
# ============================================================

print("\nGenerating individual mango explanations...")

individual_rows = []

for class_index, class_name in enumerate(class_names):

    # Select the test observation with the highest
    # predicted probability for this class.
    sample_index = int(
        np.argmax(probabilities[:, class_index])
    )

    class_values = shap_array[
        sample_index, :, class_index
    ]

    sample_features = X_transformed[sample_index]

    individual_df = pd.DataFrame({
        "feature": feature_names,
        "transformed_value": sample_features,
        "shap_value": class_values,
        "absolute_shap": np.abs(class_values)
    })

    individual_df = individual_df.sort_values(
        "absolute_shap",
        ascending=False
    )

    mango_id = (
        test_data.iloc[sample_index]["mango_id"]
        if "mango_id" in test_data.columns
        else f"TEST_SAMPLE_{sample_index}"
    )

    storage_day = (
        test_data.iloc[sample_index]["storage_day"]
        if "storage_day" in test_data.columns
        else np.nan
    )

    individual_df.insert(0, "mango_id", mango_id)
    individual_df.insert(1, "storage_day", storage_day)
    individual_df.insert(2, "explained_class", class_name)
    individual_df.insert(3, "sample_index", sample_index)

    individual_rows.append(individual_df)

    safe_name = class_name.lower().replace(" ", "_")

    individual_df.to_csv(
        OUTPUT_DIR / f"step11_individual_{safe_name}.csv",
        index=False
    )

    # Retrieve the base value for this sample and class.
    base_values = np.asarray(
        shap_explanation.base_values
    )

    if base_values.ndim == 3:
        base_value = base_values[
            sample_index, 0, class_index
        ]
    elif base_values.ndim == 2:
        base_value = base_values[
            sample_index, class_index
        ]
    elif base_values.ndim == 1:
        if len(base_values) == len(X_test):
            base_value = base_values[sample_index]
        else:
            base_value = base_values[class_index]
    else:
        base_value = base_values.item()

    explanation = shap.Explanation(
        values=class_values,
        base_values=float(base_value),
        data=sample_features,
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 9))

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        f"Individual SHAP Explanation\n"
        f"Mango: {mango_id} | "
        f"Day: {storage_day} | "
        f"Class: {class_name}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES / f"step11_waterfall_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

individual_all_df = pd.concat(
    individual_rows,
    ignore_index=True
)

individual_all_df.to_csv(
    OUTPUT_DIR / "step11_individual_explanations.csv",
    index=False
)

# ============================================================
# 16. WRITE SUMMARY REPORT
# ============================================================

print("\nWriting SHAP summary report...")

report_file = OUTPUT_DIR / "step11_shap_summary.txt"

with open(report_file, "w", encoding="utf-8") as file:

    file.write(
        "PHYSITWIN-X STEP 11\n"
        "SHAP EXPLAINABILITY FOR SPOILAGE-RISK CLASSIFICATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write("DATASET INFORMATION\n")
    file.write(f"Test observations: {len(X_test)}\n")

    if "mango_id" in test_data.columns:
        file.write(
            f"Test mangoes: {test_data['mango_id'].nunique()}\n"
        )

    file.write(
        f"Original predictors: {len(FEATURE_COLUMNS)}\n"
    )

    file.write(
        f"Transformed predictors: {len(feature_names)}\n"
    )

    file.write(
        f"Classification classes: {len(class_names)}\n\n"
    )

    file.write("SAVED ORIGINAL FEATURE ORDER\n")

    for feature in FEATURE_COLUMNS:
        file.write(f"- {feature}\n")

    file.write("\nCLASS ORDER\n")

    for index, class_name in enumerate(class_names):
        file.write(f"{index}: {class_name}\n")

    file.write("\nGLOBAL SHAP FEATURE IMPORTANCE\n")
    file.write(global_df.to_string(index=False))

    file.write("\n\nCLASS-SPECIFIC SHAP IMPORTANCE\n")
    file.write(class_importance_df.to_string(index=False))

    file.write("\n\nINTERPRETATION NOTES\n")

    file.write(
        "SHAP values explain how transformed input features "
        "contribute to the model output for each class.\n"
    )

    file.write(
        "Positive SHAP values increase the model output for "
        "the corresponding class; negative values decrease it.\n"
    )

    file.write(
        "Mean absolute SHAP values measure the average magnitude "
        "of contributions, not their direction.\n"
    )

    file.write(
        "SHAP explanations describe model behavior and do not "
        "establish causation or prove future spoilage.\n"
    )

    file.write(
        "The original risk labels may be dependent on firmness "
        "and peel-color observations used as predictors.\n"
    )

    file.write(
        "Therefore, feature contributions must be interpreted "
        "with the label-dependency assessment in Step 10D.\n"
    )

# ============================================================
# 17. FINAL OUTPUT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 OUTPUT VALIDATION")
print("=" * 70)

required_outputs = [
    OUTPUT_DIR / "step11_test_predictions.csv",
    OUTPUT_DIR / "step11_transformed_feature_mapping.csv",
    OUTPUT_DIR / "step11_shap_global_importance.csv",
    OUTPUT_DIR / "step11_shap_class_importance.csv",
    OUTPUT_DIR / "step11_shap_values.csv",
    OUTPUT_DIR / "step11_individual_explanations.csv",
    OUTPUT_DIR / "step11_shap_summary.txt",
    OUTPUT_FIGURES / "step11_global_shap_importance.png"
]

for output_file in required_outputs:

    if not output_file.exists():
        raise FileNotFoundError(
            f"Missing output: {output_file}"
        )

    if output_file.stat().st_size == 0:
        raise ValueError(
            f"Empty output: {output_file}"
        )

print("Core output validation: PASSED")

print("\n" + "=" * 70)
print("PHYSITWIN-X STEP 11 COMPLETED")
print("=" * 70)

print("\nRESULTS FOLDER:")
print(OUTPUT_DIR)

print("\nFIGURES FOLDER:")
print(OUTPUT_FIGURES)

print("\nIMPORTANT FIGURES:")
print("- Global SHAP importance")
print("- Five class-specific SHAP bar plots")
print("- Five class-specific SHAP summary plots")
print("- Five individual SHAP waterfall plots")

print("\nALL STEP 11 ANALYSES FINISHED.")
print("=" * 70)
