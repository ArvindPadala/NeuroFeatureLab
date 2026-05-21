import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "best_model.pkl")


def load_subject_json(input_path: str) -> pd.DataFrame:
    """
    Load a single subject-level feature vector from a JSON file.

    Returns
    -------
    pd.DataFrame
        One-row dataframe containing subject features.
    """

    with open(input_path, "r") as f:
        subject_data = json.load(f)

    return pd.DataFrame([subject_data])


def run_inference(
    input_path: str,
    model_path: str = DEFAULT_MODEL_PATH
):
    """
    Run inference on a single simulated subject feature vector.

    This function predicts whether the input feature pattern is closer to a
    simulated non-PTE-like or PTE-like pattern.

    This is not a clinical prediction tool.
    """

    subject_df = load_subject_json(input_path)

    model = joblib.load(model_path)

    prediction = model.predict(subject_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(subject_df)[0][1]
    else:
        probability = None

    label_map = {
        0: "non-PTE-like feature pattern",
        1: "PTE-like feature pattern"
    }

    result = {
        "prediction_label": int(prediction),
        "prediction_text": label_map[int(prediction)],
        "pte_like_probability": None if probability is None else float(probability)
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a simulated subject-level feature vector."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input subject JSON file."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model pipeline."
    )

    args = parser.parse_args()

    result = run_inference(
        input_path=args.input,
        model_path=args.model
    )

    print("\nInference result")
    print("----------------")
    print(f"Prediction: {result['prediction_text']}")

    if result["pte_like_probability"] is not None:
        print(f"PTE-like pattern probability: {result['pte_like_probability']:.3f}")

    print("\nNote: This is a simulated research-methods demo, not a clinical tool.")


if __name__ == "__main__":
    main()