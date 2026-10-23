from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data.artifact_store import resolve_processed_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR = LOCAL_PROCESSED_DIR
MODEL_FILENAME = "fraud_model.joblib"
SUMMARY_FILENAME = "evaluation_summary.json"
SPLIT_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
RUNTIME_REQUIRED_FILENAMES = (
    *SPLIT_FILENAMES.values(),
    MODEL_FILENAME,
    SUMMARY_FILENAME,
)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected data split at {path}")
    return pd.read_csv(path)


def _compute_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if y_true.size == 0 or probabilities.size != y_true.size:
        raise ValueError("Threshold selection requires matching, non-empty labels and probabilities.")
    if np.unique(y_true).size != 2:
        raise ValueError("Threshold selection requires both binary classes in the validation split.")

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        np.finfo(float).eps,
    )
    best_f1 = np.max(f1_scores)
    # Prefer the higher threshold when F1 ties, reducing alerts without sacrificing F1.
    best_index = np.flatnonzero(f1_scores == best_f1)[-1]
    return float(thresholds[best_index])


def _build_prediction_table(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    table = df.copy()
    table["transaction_id"] = [f"{split_name}-{idx:06d}" for idx in range(len(table))]
    table["model_probability"] = probabilities
    table["model_prediction"] = (probabilities >= threshold).astype(int)
    table["risk_category"] = pd.cut(
        table["model_probability"],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
        labels=["low", "medium", "high", "critical"],
        right=True,
    )
    table["ground_truth_label"] = table.get("Class", pd.Series(np.nan, index=table.index))
    return table


def _summarize_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    pr_auc = average_precision_score(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "selected_threshold": float(threshold),
        "flagged_transactions": int(predictions.sum()),
        "actual_fraud_cases": int(y_true.sum()),
    }


def load_split_data(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    processed_dir = processed_dir or resolve_processed_dir(SPLIT_FILENAMES.values())
    return {
        name: _load_csv(processed_dir / filename)
        for name, filename in SPLIT_FILENAMES.items()
    }


def ensure_data_available(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    split_data = load_split_data(processed_dir)
    expected_columns = list(split_data["train"].columns)
    for name, frame in split_data.items():
        if "Class" not in frame.columns:
            raise ValueError(f"Missing Class column in {name} split.")
        if list(frame.columns) != expected_columns:
            raise ValueError(f"Columns in {name} split do not match the training split.")
        if frame.empty or frame["Class"].nunique() != 2:
            raise ValueError(f"{name} split must contain rows from both binary classes.")
    return split_data


def train_baseline_model(save_artifacts: bool = True) -> dict[str, Any]:
    split_data = ensure_data_available()
    train_df = split_data["train"].copy()
    validation_df = split_data["validation"].copy()
    test_df = split_data["test"].copy()

    feature_columns = [col for col in train_df.columns if col != "Class"]
    X_train = train_df[feature_columns]
    y_train = train_df["Class"].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    X_validation = scaler.transform(validation_df[feature_columns])
    validation_scores = model.predict_proba(X_validation)[:, 1]
    val_threshold = _compute_threshold(validation_df["Class"].astype(int).to_numpy(), validation_scores)

    X_test = scaler.transform(test_df[feature_columns])
    test_scores = model.predict_proba(X_test)[:, 1]

    validation_summary = _summarize_metrics(
        validation_df["Class"].astype(int).to_numpy(),
        validation_scores,
        val_threshold,
    )
    test_summary = _summarize_metrics(
        test_df["Class"].astype(int).to_numpy(),
        test_scores,
        val_threshold,
    )

    validation_table = _build_prediction_table(
        validation_df,
        validation_scores,
        val_threshold,
        "validation",
    )
    test_table = _build_prediction_table(
        test_df,
        test_scores,
        val_threshold,
        "test",
    )

    artifact = {
        "model_name": "logistic_regression_balanced",
        "selected_threshold": val_threshold,
        "feature_columns": feature_columns,
        "training_rows": int(len(train_df)),
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "validation_table": validation_table,
        "test_table": test_table,
        "scaler": scaler,
        "model": model,
        "raw_dataset_rows": int(len(pd.concat([train_df, validation_df, test_df], ignore_index=True))),
    }

    if save_artifacts:
        processed_dir = LOCAL_PROCESSED_DIR
        processed_dir.mkdir(parents=True, exist_ok=True)
        model_path = processed_dir / MODEL_FILENAME
        joblib.dump({"model": model, "scaler": scaler, "threshold": val_threshold}, model_path)
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        summary_payload = {
            "model_name": artifact["model_name"],
            "selected_threshold": val_threshold,
            "training_rows": int(len(train_df)),
            "feature_columns": feature_columns,
            "model_sha256": model_sha256,
            "validation": validation_summary,
            "test": test_summary,
        }
        (processed_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )

    return artifact


def load_trained_model(
    processed_dir: Path | None = None,
    require_splits: bool = False,
) -> dict[str, Any]:
    required_files = (
        RUNTIME_REQUIRED_FILENAMES
        if require_splits
        else (MODEL_FILENAME, SUMMARY_FILENAME)
    )
    processed_dir = processed_dir or resolve_processed_dir(required_files)
    model_path = processed_dir / MODEL_FILENAME
    summary_path = processed_dir / SUMMARY_FILENAME
    try:
        loaded = joblib.load(model_path)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        ValueError,
        KeyError,
        AttributeError,
        ImportError,
        TypeError,
        IndexError,
    ) as exc:
        raise ValueError(
            f"Could not deserialize model artifact at {model_path}. "
            "Verify it is the matching fraud_model.joblib and upload it again if corrupted."
        ) from exc
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read evaluation summary at {summary_path}. "
            "Verify evaluation_summary.json and retrain the baseline if it is corrupted."
        ) from exc
    if "model" not in loaded or "scaler" not in loaded or "threshold" not in loaded:
        raise ValueError(f"Model artifact at {model_path} is incomplete. Retrain the baseline model.")
    model = loaded["model"]
    scaler = loaded["scaler"]
    feature_columns = summary.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("Evaluation summary has no feature schema. Retrain the baseline model.")
    expected_count = len(feature_columns)
    if (
        getattr(model, "n_features_in_", None) != expected_count
        or getattr(scaler, "n_features_in_", None) != expected_count
        or list(getattr(model, "classes_", [])) != [0, 1]
    ):
        raise ValueError("Model artifact feature schema or fraud class does not match its evaluation summary.")
    for fitted_names in (
        getattr(model, "feature_names_in_", None),
        getattr(scaler, "feature_names_in_", None),
    ):
        if fitted_names is not None and list(fitted_names) != feature_columns:
            raise ValueError("Model feature ordering does not match its evaluation summary.")
    artifact_threshold = float(loaded["threshold"])
    summary_threshold = float(summary.get("selected_threshold", float("nan")))
    if not np.isfinite(artifact_threshold) or artifact_threshold != summary_threshold:
        raise ValueError("Model threshold does not match its evaluation summary. Retrain the baseline model.")
    expected_hash = summary.get("model_sha256")
    if not expected_hash:
        raise ValueError("Evaluation summary has no model checksum. Retrain and publish matching artifacts.")
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Model artifact checksum does not match the evaluation summary.")
    output = {
        "model": model,
        "scaler": scaler,
        "selected_threshold": artifact_threshold,
        "feature_columns": feature_columns,
        "summary": summary,
    }
    return output


def build_evaluation_frame() -> pd.DataFrame:
    processed_dir = resolve_processed_dir(RUNTIME_REQUIRED_FILENAMES)
    split_data = ensure_data_available(processed_dir)
    model_artifact = load_trained_model(processed_dir=processed_dir)
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    threshold = model_artifact["selected_threshold"]
    feature_columns = model_artifact["feature_columns"]
    if any(column not in split_data["train"].columns for column in feature_columns):
        raise ValueError("Trained model feature schema does not match the downloaded transaction splits.")

    frames = []
    for split_name, frame in split_data.items():
        subset = frame.copy()
        X = scaler.transform(subset[feature_columns])
        probabilities = model.predict_proba(X)[:, 1]
        subset["split"] = split_name
        subset["transaction_id"] = [f"{split_name}-{index:06d}" for index in range(len(subset))]
        subset["model_probability"] = probabilities
        subset["model_prediction"] = (probabilities >= threshold).astype(int)
        subset["risk_category"] = pd.cut(
            subset["model_probability"],
            bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
            labels=["low", "medium", "high", "critical"],
            right=True,
        )
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def get_feature_contributions(model: LogisticRegression, scaler, row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    """Return signed per-feature contributions to the fraud-class logit."""
    values = row[feature_columns].to_numpy(dtype=float)
    classes = np.asarray(model.classes_)
    fraud_class_index = np.flatnonzero(classes == 1)
    if fraud_class_index.size != 1:
        raise ValueError("Feature contributions require a model with fraud class label 1.")
    if model.coef_.shape[0] != 1 or len(classes) != 2:
        raise ValueError("Feature contributions currently support binary linear models only.")
    if model.coef_.shape[1] != len(feature_columns):
        raise ValueError("Model coefficient order does not match the requested feature columns.")
    if getattr(scaler, "n_features_in_", len(feature_columns)) != len(feature_columns):
        raise ValueError("Scaler feature order does not match the requested feature columns.")
    for fitted_names in (
        getattr(model, "feature_names_in_", None),
        getattr(scaler, "feature_names_in_", None),
    ):
        if fitted_names is not None and list(fitted_names) != feature_columns:
            raise ValueError("Requested feature order does not match the fitted model and scaler.")

    feature_frame = pd.DataFrame([values], columns=feature_columns)
    scaled_values = np.asarray(scaler.transform(feature_frame))[0]
    # For sklearn's binary linear estimators, coef_[0] describes classes_[1].
    # Reverse its sign when fraud (label 1) is classes_[0].
    orientation = 1.0 if fraud_class_index[0] == 1 else -1.0
    coefficients = orientation * model.coef_[0]
    contributions = coefficients * scaled_values
    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "value": values,
            "scaled_value": scaled_values,
            "coefficient": coefficients,
            "contribution": contributions,
            "contribution_score": np.abs(contributions),
        }
    ).sort_values("contribution_score", ascending=False)
    return ranking.reset_index(drop=True)
