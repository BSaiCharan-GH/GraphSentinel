from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from huggingface_hub.errors import HfHubHTTPError
from sklearn.linear_model import LogisticRegression

from src.data import artifact_store
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


def test_prediction_frame_contains_expected_columns(
    trained_artifacts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_requirements: list[tuple[str, ...]] = []
    resolve_processed_dir = fraud_model.resolve_processed_dir

    def record_runtime_bundle(required: tuple[str, ...]) -> Path:
        resolved_requirements.append(tuple(required))
        return resolve_processed_dir(required)

    monkeypatch.setattr(fraud_model, "resolve_processed_dir", record_runtime_bundle)
    frame = build_evaluation_frame()
    expected = {"transaction_id", "model_probability", "model_prediction", "risk_category", "Class", "split"}
    assert expected.issubset(set(frame.columns))
    assert len(frame) > 0
    assert set(frame["Class"].dropna().unique()).issubset({0, 1})
    assert resolved_requirements == [fraud_model.RUNTIME_REQUIRED_FILENAMES]


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
    monkeypatch.setenv("GRAPHSENTINEL_PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.delenv("GRAPHSENTINEL_ARTIFACT_REPO", raising=False)
    monkeypatch.delenv("GRAPHSENTINEL_HF_TOKEN", raising=False)
    with pytest.raises(FileNotFoundError, match=r"\[artifacts\]\.repo_id"):
        fraud_model.load_trained_model()


def test_configured_external_artifact_directory_is_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_processed = tmp_path / "remote" / "processed"
    remote_processed.mkdir(parents=True)
    (remote_processed / "train.csv").write_text("Class\n0\n1\n", encoding="utf-8")
    monkeypatch.setattr(artifact_store, "_artifact_repo_id", None)
    monkeypatch.setattr(artifact_store, "_artifact_token", None)
    monkeypatch.setattr(artifact_store, "_artifact_revision", None)
    monkeypatch.setattr(artifact_store, "_artifact_local_dir", None)
    monkeypatch.setenv("GRAPHSENTINEL_PROCESSED_DIR", str(tmp_path / "local"))
    artifact_store.configure_from_mapping(
        {
            "artifacts": {
                "repo_id": "https://huggingface.co/datasets/example/private-dataset",
                "token": "not-a-real-token",
            }
        }
    )
    monkeypatch.setattr(
        artifact_store,
        "_download_snapshot",
        lambda repo_id, revision, token, required: remote_processed,
    )

    resolved = artifact_store.resolve_processed_dir(["train.csv"])
    assert resolved == remote_processed


def test_external_artifact_download_is_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import huggingface_hub

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        processed = tmp_path / "snapshot" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "train.csv").write_text("Class\n0\n1\n", encoding="utf-8")
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    artifact_store._download_snapshot_cached.cache_clear()
    try:
        first = artifact_store._download_snapshot(
            "example/private-dataset",
            "main",
            "test-token",
            ("train.csv",),
        )
        second = artifact_store._download_snapshot(
            "example/private-dataset",
            "main",
            "test-token",
            ("train.csv",),
        )
    finally:
        artifact_store._download_snapshot_cached.cache_clear()

    assert first == second == tmp_path / "snapshot" / "processed"
    assert len(calls) == 1
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["allow_patterns"] == ["processed/train.csv"]


def test_authentication_failure_has_safe_actionable_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub
    import httpx2

    response = httpx2.Response(
        status_code=401,
        request=httpx2.Request("GET", "https://huggingface.co/api/datasets/example/private-dataset"),
    )

    def denied_download(**kwargs: object) -> str:
        raise HfHubHTTPError("unauthorized", response=response)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", denied_download)
    artifact_store._download_snapshot_cached.cache_clear()

    with pytest.raises(artifact_store.ArtifactStoreError) as error:
        artifact_store._download_snapshot(
            "example/private-dataset",
            "main",
            "never-log-this-token",
            ("train.csv",),
        )

    message = str(error.value)
    assert "denied access" in message
    assert "processed/train.csv" in message
    assert "never-log-this-token" not in message
    assert error.value.__cause__ is None
    artifact_store._download_snapshot_cached.cache_clear()


def test_missing_remote_files_report_expected_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(artifact_store, "_artifact_repo_id", "example/private-dataset")
    monkeypatch.setattr(artifact_store, "_artifact_token", "test-token")
    monkeypatch.setattr(artifact_store, "_artifact_revision", "main")
    monkeypatch.setenv("GRAPHSENTINEL_PROCESSED_DIR", str(tmp_path / "local"))
    monkeypatch.setattr(
        artifact_store,
        "_download_snapshot",
        lambda *args: tmp_path / "snapshot" / "processed",
    )

    with pytest.raises(artifact_store.ArtifactStoreError, match="processed/train.csv"):
        artifact_store.resolve_processed_dir(["train.csv"])


def test_missing_private_repository_token_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[artifacts\]\.token is missing"):
        artifact_store.configure_from_mapping(
            {"artifacts": {"repo_id": "example/private-dataset"}}
        )


def test_configuration_change_invalidates_download_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    monkeypatch.delenv("GRAPHSENTINEL_ARTIFACT_REPO", raising=False)
    monkeypatch.delenv("GRAPHSENTINEL_HF_TOKEN", raising=False)
    monkeypatch.delenv("GRAPHSENTINEL_ARTIFACT_REVISION", raising=False)
    monkeypatch.setattr(artifact_store, "_artifact_repo_id", None)
    monkeypatch.setattr(artifact_store, "_artifact_token", None)
    monkeypatch.setattr(artifact_store, "_artifact_revision", None)
    monkeypatch.setattr(artifact_store, "_artifact_local_dir", None)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(artifact_store.PROJECT_ROOT),
    )
    artifact_store.configure_artifact_store("example/first", "token-one", "main")
    artifact_store._download_snapshot_cached(
        "example/first", "main", "token-one", ("train.csv",)
    )
    assert artifact_store._download_snapshot_cached.cache_info().currsize == 1
    previous_version = artifact_store.configuration_version()

    artifact_store.configure_artifact_store("example/second", "token-two", "main")

    assert artifact_store.configuration_version() == previous_version + 1
    assert artifact_store._download_snapshot_cached.cache_info().currsize == 0


def test_processed_directory_change_invalidates_app_initialization_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(artifact_store, "_artifact_repo_id", None)
    monkeypatch.setattr(artifact_store, "_artifact_token", None)
    monkeypatch.setattr(artifact_store, "_artifact_revision", None)
    monkeypatch.setattr(artifact_store, "_artifact_local_dir", None)
    monkeypatch.setenv("GRAPHSENTINEL_PROCESSED_DIR", str(tmp_path / "first"))
    artifact_store.configure_from_mapping({})
    first_version = artifact_store.configuration_version()

    monkeypatch.setenv("GRAPHSENTINEL_PROCESSED_DIR", str(tmp_path / "second"))
    artifact_store.configure_from_mapping({})

    assert artifact_store.configuration_version() == first_version + 1


def test_relative_processed_directory_is_resolved_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHSENTINEL_PROCESSED_DIR", "deployment-data")
    assert artifact_store._configured_local_dir() == (
        artifact_store.PROJECT_ROOT / "deployment-data"
    ).resolve()


@pytest.mark.parametrize(
    "repo_id",
    [
        "https://example.com/datasets/owner/repo",
        "https://huggingface.co/owner/repo",
        "https://huggingface.co/datasets/owner/repo/tree/main",
    ],
)
def test_invalid_hugging_face_repository_urls_are_rejected(repo_id: str) -> None:
    with pytest.raises(ValueError, match="owner/repository or a dataset URL"):
        artifact_store.configure_artifact_store(repo_id=repo_id)


def test_evaluation_summary_matches_current_test_split(trained_artifacts: None) -> None:
    model_artifact = load_trained_model()
    test_frame = build_evaluation_frame().query("split == 'test'")
    actual = fraud_model._summarize_metrics(
        test_frame["Class"].astype(int).to_numpy(),
        test_frame["model_probability"].to_numpy(dtype=float),
        model_artifact["selected_threshold"],
    )
    recorded = model_artifact["summary"]["test"]
    for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert recorded[metric] == pytest.approx(actual[metric])


def test_small_model_artifact_loads_without_transaction_splits(
    trained_artifacts: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_split_load() -> None:
        raise AssertionError("Model artifact loading should not require transaction splits.")

    monkeypatch.setattr(fraud_model, "ensure_data_available", unexpected_split_load)
    artifact = load_trained_model()
    assert len(artifact["feature_columns"]) == artifact["model"].n_features_in_
    assert artifact["model"].classes_.tolist() == [0, 1]


def test_corrupted_model_artifact_has_actionable_error(
    trained_artifacts: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "processed"
    artifact_dir.mkdir()
    shutil.copy(
        fraud_model.PROCESSED_DIR / fraud_model.SUMMARY_FILENAME,
        artifact_dir / fraud_model.SUMMARY_FILENAME,
    )
    (artifact_dir / fraud_model.MODEL_FILENAME).write_bytes(b"not a joblib artifact")
    monkeypatch.setattr(fraud_model, "resolve_processed_dir", lambda required: artifact_dir)

    with pytest.raises(ValueError, match="Could not deserialize model artifact"):
        load_trained_model()


def test_invalid_evaluation_json_has_actionable_error(
    trained_artifacts: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "processed"
    artifact_dir.mkdir()
    shutil.copy(
        fraud_model.PROCESSED_DIR / fraud_model.MODEL_FILENAME,
        artifact_dir / fraud_model.MODEL_FILENAME,
    )
    (artifact_dir / fraud_model.SUMMARY_FILENAME).write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(fraud_model, "resolve_processed_dir", lambda required: artifact_dir)

    with pytest.raises(ValueError, match="Could not read evaluation summary"):
        load_trained_model()


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
