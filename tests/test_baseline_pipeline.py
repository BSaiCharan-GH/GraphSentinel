from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.graph.graph_builder import build_similarity_graph
from src.models import fraud_model
from src.models.fraud_model import (
    build_evaluation_frame,
    get_feature_contributions,
    load_trained_model,
)


def test_loaded_model_artifact_has_expected_keys(trained_artifacts: None) -> None:
    artifact = load_trained_model()
    assert "model" in artifact
    assert "scaler" in artifact
    assert "selected_threshold" in artifact
    assert "feature_columns" in artifact


def test_prediction_frame_contains_expected_columns(trained_artifacts: None) -> None:
    frame = build_evaluation_frame()
    expected = {"transaction_id", "model_probability", "model_prediction", "risk_category", "Class", "split"}
    assert expected.issubset(set(frame.columns))
    assert len(frame) > 0
    assert set(frame["Class"].dropna().unique()).issubset({0, 1})


def test_validation_split_is_class_balanced_enough(processed_splits: None) -> None:
    train = pd.read_csv(fraud_model.PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(fraud_model.PROCESSED_DIR / "validation.csv")
    assert train["Class"].nunique() == 2
    assert validation["Class"].nunique() == 2
    assert len(validation) > 0


def test_similarity_graph_uses_feature_space_only(trained_artifacts: None) -> None:
    artifact = load_trained_model()
    frame = build_evaluation_frame().head(200)
    graph = build_similarity_graph(frame, artifact["feature_columns"], k_neighbors=5, max_nodes=100)
    assert graph["nodes"]
    assert graph["edges"]
    assert all(edge["source"] != edge["target"] for edge in graph["edges"])
    assert all(edge["source"] in {node["id"] for node in graph["nodes"]} for edge in graph["edges"])
    assert all(edge["target"] in {node["id"] for node in graph["nodes"]} for edge in graph["edges"])


def test_similarity_graph_excludes_target_label_from_distances() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [f"tx-{index}" for index in range(12)],
            "feature_a": np.arange(12, dtype=float),
            "feature_b": np.arange(12, dtype=float) ** 2,
            "Class": [0, 1] * 6,
        }
    )
    changed_labels = frame.copy()
    changed_labels["Class"] = 1 - changed_labels["Class"]

    graph = build_similarity_graph(
        frame,
        ["feature_a", "feature_b", "Class"],
        k_neighbors=3,
        max_nodes=12,
    )
    changed_graph = build_similarity_graph(
        changed_labels,
        ["feature_a", "feature_b", "Class"],
        k_neighbors=3,
        max_nodes=12,
    )
    assert graph["edges"] == changed_graph["edges"]
    assert np.allclose(
        [(node["x"], node["y"]) for node in graph["nodes"]],
        [(node["x"], node["y"]) for node in changed_graph["nodes"]],
    )


def test_threshold_selection_chooses_maximum_validation_f1() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    threshold = fraud_model._compute_threshold(labels, probabilities)
    predicted = probabilities >= threshold
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    precision = true_positive / int(predicted.sum())
    recall = true_positive / int(labels.sum())
    selected_f1 = 2 * precision * recall / (precision + recall)
    assert selected_f1 == pytest.approx(0.8)


def test_missing_model_artifacts_do_not_trigger_implicit_training(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fraud_model, "MODEL_PATH", tmp_path / "missing-model.joblib")
    monkeypatch.setattr(fraud_model, "SUMMARY_PATH", tmp_path / "missing-summary.json")
    with pytest.raises(FileNotFoundError, match="train_baseline.py"):
        fraud_model.load_trained_model()


@pytest.mark.parametrize(
    ("values", "feature", "expected"),
    [
        ([1.0, 0.0, 0.0], "first", 2.0),
        ([0.0, 2.0, 0.0], "second", -6.0),
        ([0.0, 0.0, 1.0], "third", 0.0),
    ],
    ids=["positive", "negative", "zero"],
)
def test_signed_feature_contributions_preserve_direction_and_rank(
    values: list[float],
    feature: str,
    expected: float,
) -> None:
    class IdentityScaler:
        n_features_in_ = 3

        def transform(self, features: np.ndarray) -> np.ndarray:
            return features

    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[2.0, -3.0, 0.0]])
    model.n_features_in_ = 3
    row = pd.Series(dict(zip(["first", "second", "third"], values)))

    contributions = get_feature_contributions(
        model,
        IdentityScaler(),
        row,
        ["first", "second", "third"],
    )
    by_feature = contributions.set_index("feature")
    by_feature = contributions.set_index("feature")
    assert by_feature.loc[feature, "contribution"] == pytest.approx(expected)
    assert contributions["contribution_score"].is_monotonic_decreasing


def test_signed_feature_contribution_respects_fraud_class_orientation() -> None:
    class IdentityScaler:
        n_features_in_ = 1

        def transform(self, features: np.ndarray) -> np.ndarray:
            return features

    model = LogisticRegression()
    model.classes_ = np.array([1, 0])
    model.coef_ = np.array([[2.0]])
    model.n_features_in_ = 1
    contribution = get_feature_contributions(
        model,
        IdentityScaler(),
        pd.Series({"feature": 3.0}),
        ["feature"],
    )
    assert contribution.loc[0, "contribution"] == pytest.approx(-6.0)


def test_feature_contributions_use_transformed_values_and_fitted_order() -> None:
    class TransformingScaler:
        n_features_in_ = 2
        feature_names_in_ = np.array(["first", "second"])

        def transform(self, features: pd.DataFrame) -> np.ndarray:
            assert list(features.columns) == ["first", "second"]
            return (features.to_numpy() - np.array([1.0, 2.0])) * np.array([2.0, 3.0])

    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[2.0, -1.0]])
    model.feature_names_in_ = np.array(["first", "second"])
    model.n_features_in_ = 2
    contribution = get_feature_contributions(
        model,
        TransformingScaler(),
        pd.Series({"first": 3.0, "second": 1.0}),
        ["first", "second"],
    )
    by_feature = contribution.set_index("feature")
    assert by_feature.loc["first", "scaled_value"] == pytest.approx(4.0)
    assert by_feature.loc["first", "contribution"] == pytest.approx(8.0)
    assert by_feature.loc["second", "scaled_value"] == pytest.approx(-3.0)
    assert by_feature.loc["second", "contribution"] == pytest.approx(3.0)
