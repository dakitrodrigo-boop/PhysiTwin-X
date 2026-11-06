
# ============================================================
# PHYSITWIN-X STEP 11A
# INSPECT SAVED SPOILAGE-RISK CLASSIFICATION MODEL
# ============================================================

from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    ROOT / "results" /
    "physitwin_x_spoilage_risk_model.joblib"
)

print("=" * 70)
print("PHYSITWIN-X STEP 11A")
print("SAVED MODEL INSPECTION")
print("=" * 70)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )

print("\nLoading saved model...")

saved_object = joblib.load(MODEL_FILE)

print("\nMODEL OBJECT TYPE:")
print(type(saved_object))

if isinstance(saved_object, dict):

    print("\nSAVED MODEL KEYS:")

    for key in saved_object:
        print(f"- {key}")

    print("\nFEATURE COLUMNS SAVED IN MODEL:")

    features = saved_object.get("feature_columns")

    if features is None:
        print("No feature_columns entry found.")
    else:
        print("Number of features:", len(features))

        for index, feature in enumerate(features, start=1):
            print(f"{index}. {feature}")

    print("\nSAVED CLASS ORDER:")
    print(saved_object.get("classes"))

    print("\nLABEL ENCODER:")

    encoder = saved_object.get("label_encoder")

    if encoder is not None:
        print("Encoder type:", type(encoder))
        print("Encoder classes:", encoder.classes_)
    else:
        print("No label encoder found.")

    print("\nMODEL:")

    model = saved_object.get("model")

    if model is not None:
        print("Model type:", type(model))

        if hasattr(model, "steps"):
            print("\nPIPELINE STEPS:")

            for name, component in model.steps:
                print(name, ":", type(component))

        if hasattr(model, "feature_names_in_"):
            print("\nMODEL INPUT FEATURE NAMES:")
            print(model.feature_names_in_)

        if hasattr(model, "get_booster"):
            booster = model.get_booster()

            print("\nXGBOOST BOOSTER FEATURE NAMES:")
            print(booster.feature_names)

else:

    print("\nSaved object is not a dictionary.")
    print("Inspecting object attributes:")

    print(dir(saved_object))

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)
