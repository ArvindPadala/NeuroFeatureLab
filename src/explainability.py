import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def generate_permutation_importance(
    data_path: str = "data/simulated_pte_features.csv",
    model_path: str = "models/best_model.pkl",
    output_path: str = "outputs/feature_importance.png",
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
    print("- outputs/feature_importance.png")