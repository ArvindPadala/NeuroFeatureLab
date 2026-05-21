import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "simulated_pte_features.csv")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
DEFAULT_MODELS_DIR = str(PROJECT_ROOT / "models")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def load_dataset(data_path: str = DEFAULT_DATA_PATH):
    """
    Load the simulated imaging-derived feature dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    feature_names : list
        Feature column names.
    """

    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    drop_cols = ["label"]

    if "subject_id" in df.columns:
        drop_cols.append("subject_id")

    X = df.drop(columns=drop_cols)
    y = df["label"]

    feature_names = X.columns.tolist()

    return X, y, feature_names


def get_models():
    """
    Define models for comparison.

    These models are selected to demonstrate classical ML comparison
    in a research-style workflow.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Linear SVM": SVC(
            kernel="linear",
            probability=True,
            random_state=42
        ),
        "RBF Kernel SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=42
        )
    }

    return models


def evaluate_models(
    X,
    y,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    models_dir: str = DEFAULT_MODELS_DIR
):
    """
    Train/test split + cross-validation model comparison.

    Saves:
    - model comparison CSV
    - ROC curve for best model
    - confusion matrix for best model
    - best model pipeline
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = get_models()
    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        cv_auc_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = pipeline.decision_function(X_test)

        test_auc = roc_auc_score(y_test, y_score)

        results.append({
            "model": model_name,
            "test_auc": test_auc,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "cv_auc_mean": cv_auc_scores.mean(),
            "cv_auc_std": cv_auc_scores.std()
        })

        trained_pipelines[model_name] = {
            "pipeline": pipeline,
            "test_auc": test_auc,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_score": y_score
        }

    results_df = pd.DataFrame(results).sort_values(
        by="test_auc",
        ascending=False
    )

    results_path = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]["pipeline"]

    joblib.dump(best_pipeline, os.path.join(models_dir, "best_model.pkl"))

    # Also save scaler separately for transparency, although the full pipeline is enough
    joblib.dump(best_pipeline.named_steps["scaler"], os.path.join(models_dir, "scaler.pkl"))

    # ROC curve
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        trained_pipelines[best_model_name]["y_test"],
        trained_pipelines[best_model_name]["y_score"]
    )
    plt.title(f"ROC Curve - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        trained_pipelines[best_model_name]["y_test"],
        trained_pipelines[best_model_name]["y_pred"],
        display_labels=["non-PTE-like", "PTE-like"]
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    return results_df, best_model_name


if __name__ == "__main__":
    X, y, feature_names = load_dataset()
    results_df, best_model_name = evaluate_models(X, y)

    print("Model training and evaluation complete.")
    print("\nModel comparison:")
    print(results_df)
    print(f"\nBest model: {best_model_name}")
    print("\nSaved outputs:")
    print(f"- {os.path.join(DEFAULT_OUTPUT_DIR, 'model_comparison.csv')}")
    print(f"- {os.path.join(DEFAULT_OUTPUT_DIR, 'roc_curve.png')}")
    print(f"- {os.path.join(DEFAULT_OUTPUT_DIR, 'confusion_matrix.png')}")
    print(f"- {os.path.join(DEFAULT_MODELS_DIR, 'best_model.pkl')}")
    print(f"- {os.path.join(DEFAULT_MODELS_DIR, 'scaler.pkl')}")