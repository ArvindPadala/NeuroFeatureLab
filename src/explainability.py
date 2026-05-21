import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "simulated_pte_features.csv")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "best_model.pkl")
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "feature_importance.png")

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def generate_permutation_importance(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    top_n: int = 12
):
    """
    Generate permutation-based feature importance for the trained model.

    This method is model-agnostic and works for SVMs, random forests,
    logistic regression, and neural networks.
    """

    df = pd.read_csv(data_path)

    drop_cols = ["label"]

    if "subject_id" in df.columns:
        drop_cols.append("subject_id")

    X = df.drop(columns=drop_cols)
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = joblib.load(model_path)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc"
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_features = importance_df.head(top_n).sort_values(
        by="importance_mean",
        ascending=True
    )

    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance_mean"])
    plt.xlabel("Mean permutation importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return importance_df


if __name__ == "__main__":
    importance_df = generate_permutation_importance()

    print("Feature importance generated successfully.")
    print("\nTop features:")
    print(importance_df.head(12))
    print("\nSaved output:")
    print(f"- {DEFAULT_OUTPUT_PATH}")