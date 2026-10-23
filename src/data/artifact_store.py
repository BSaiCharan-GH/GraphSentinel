from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

_artifact_repo_id: str | None = None
_artifact_token: str | None = None
_artifact_revision: str | None = None
_configuration_version = 0


class ArtifactStoreError(FileNotFoundError):
    """Raised when configured external artifacts cannot be resolved."""


def _normalize_repo_id(repo_id: str) -> str:
    value = repo_id.strip()
    if not value.startswith(("https://", "http://")):
        return value.strip("/")

    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.hostname not in {"huggingface.co", "www.huggingface.co"}
        or len(path_parts) != 3
        or path_parts[0] != "datasets"
    ):
        raise ValueError(
            "Hugging Face repo_id must be owner/repository or a dataset URL "
            "such as https://huggingface.co/datasets/owner/repository."
        )
    return "/".join(path_parts[1:])


def configure_artifact_store(
    repo_id: str | None = None,
    token: str | None = None,
    revision: str | None = None,
) -> None:
    global _artifact_repo_id, _artifact_token, _artifact_revision, _configuration_version
    configured_repo_id = repo_id or os.getenv("GRAPHSENTINEL_ARTIFACT_REPO")
    next_configuration = (
        _normalize_repo_id(configured_repo_id) if configured_repo_id else None,
        token or os.getenv("GRAPHSENTINEL_HF_TOKEN"),
        revision or os.getenv("GRAPHSENTINEL_ARTIFACT_REVISION"),
    )
    current_configuration = (_artifact_repo_id, _artifact_token, _artifact_revision)
    if next_configuration != current_configuration:
        _artifact_repo_id, _artifact_token, _artifact_revision = next_configuration
        _configuration_version += 1
        _download_snapshot.cache_clear()


def configuration_version() -> int:
    return _configuration_version


def _configured_local_dir() -> Path | None:
    configured = os.getenv("GRAPHSENTINEL_PROCESSED_DIR")
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@lru_cache(maxsize=1)
def _download_snapshot(
    repo_id: str,
    revision: str | None,
    token: str | None,
    required_files: tuple[str, ...],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError
    except ImportError as exc:
        raise ArtifactStoreError(
            "External artifact storage is configured, but huggingface_hub is not installed. "
            "Install project dependencies from requirements.txt."
        ) from exc

    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            allow_patterns=[f"processed/{name}" for name in required_files],
        )
    except (EntryNotFoundError, RevisionNotFoundError, OSError, ValueError) as exc:
        raise ArtifactStoreError(
            "Could not load GraphSentinel artifacts from the configured Hugging Face "
            f"dataset repository '{repo_id}'. Check the repository ID, token permissions, "
            "revision, and network availability in Streamlit Secrets."
        ) from exc
    return Path(snapshot) / "processed"


def resolve_processed_dir(required_files: Iterable[str] = ()) -> Path:
    required = tuple(sorted(set(required_files)))
    configured_dir = _configured_local_dir()
    local_dir = configured_dir or LOCAL_PROCESSED_DIR
    if all((local_dir / name).is_file() for name in required):
        return local_dir

    repo_id = _artifact_repo_id or os.getenv("GRAPHSENTINEL_ARTIFACT_REPO")
    token = _artifact_token or os.getenv("GRAPHSENTINEL_HF_TOKEN")
    revision = _artifact_revision or os.getenv("GRAPHSENTINEL_ARTIFACT_REVISION")
    if repo_id:
        repo_id = _normalize_repo_id(repo_id)
        snapshot_dir = _download_snapshot(repo_id, revision, token, required)
        missing = [name for name in required if not (snapshot_dir / name).is_file()]
        if not missing:
            return snapshot_dir
        raise ArtifactStoreError(
            "The configured artifact repository is missing required files: "
            + ", ".join(missing)
            + ". Upload the files under its processed/ directory."
        )

    missing = [name for name in required if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required GraphSentinel files are missing from {local_dir}: "
            + ", ".join(missing)
            + ". Add a local processed-data directory with GRAPHSENTINEL_PROCESSED_DIR "
            "or configure the private Hugging Face dataset repository in Streamlit Secrets."
        )
    return local_dir


def configure_from_mapping(secrets: Mapping[str, object]) -> None:
    """Read the optional [artifacts] Streamlit Secrets table."""
    values = secrets.get("artifacts", {})
    if not isinstance(values, Mapping):
        raise ValueError("Streamlit Secrets [artifacts] must be a table.")
    configure_artifact_store(
        repo_id=str(values["repo_id"]) if values.get("repo_id") else None,
        token=str(values["token"]) if values.get("token") else None,
        revision=str(values["revision"]) if values.get("revision") else None,
    )
