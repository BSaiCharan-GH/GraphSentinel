from __future__ import annotations

import pandas as pd

from src.graph.graph_builder import build_similarity_graph
from src.models.fraud_model import build_evaluation_frame, load_trained_model


def test_loaded_model_artifact_has_expected_keys() -> None:
    artifact = load_trained_model()
    assert "model" in artifact
    assert "scaler" in artifact
    assert "selected_threshold" in artifact
    assert "feature_columns" in artifact


def test_prediction_frame_contains_expected_columns() -> None:
    frame = build_evaluation_frame()
    expected = {"transaction_id", "model_probability", "model_prediction", "risk_category", "Class", "split"}
    assert expected.issubset(set(frame.columns))
    assert len(frame) > 0
    assert set(frame["Class"].dropna().unique()).issubset({0, 1})


def test_validation_split_is_class_balanced_enough() -> None:
    train = pd.read_csv("data/processed/train.csv")
    validation = pd.read_csv("data/processed/validation.csv")
    assert train["Class"].nunique() == 2
    assert validation["Class"].nunique() == 2
    assert len(validation) > 0


def test_similarity_graph_uses_feature_space_only() -> None:
    artifact = load_trained_model()
    frame = build_evaluation_frame().head(200)
    graph = build_similarity_graph(frame, artifact["feature_columns"], k_neighbors=5, max_nodes=100)
    assert graph["nodes"]
    assert graph["edges"]
    assert all(edge["source"] != edge["target"] for edge in graph["edges"])
    assert all(edge["source"] in {node["id"] for node in graph["nodes"]} for edge in graph["edges"])
    assert all(edge["target"] in {node["id"] for node in graph["nodes"]} for edge in graph["edges"])
