from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

_artifact_repo_id: str | None = None
_artifact_token: str | None = None
_artifact_revision: str | None = None
_artifact_local_dir: str | None = None
_configuration_version = 0
_DOWNLOAD_LOCK = threading.Lock()


class ArtifactStoreError(FileNotFoundError):
    """Raised when configured external artifacts cannot be resolved."""


def _normalize_repo_id(repo_id: str) -> str:
    value = repo_id.strip()
    if not value.startswith(("https://", "http://")):
        repo_parts = value.strip("/").split("/")
        if len(repo_parts) != 2 or not all(repo_parts):
            raise ValueError(
                "Hugging Face repo_id must use owner/repository format, "
                "for example BSaiCharan-HF/graphsentinel-data."
            )
        return "/".join(repo_parts)

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
    global _artifact_repo_id, _artifact_token, _artifact_revision
    global _artifact_local_dir, _configuration_version
    configured_repo_id = repo_id or os.getenv("GRAPHSENTINEL_ARTIFACT_REPO")
    local_dir = _configured_local_dir()
    next_configuration = (
        _normalize_repo_id(configured_repo_id) if configured_repo_id else None,
        token or os.getenv("GRAPHSENTINEL_HF_TOKEN"),
        revision or os.getenv("GRAPHSENTINEL_ARTIFACT_REVISION"),
        str(local_dir) if local_dir else None,
    )
    with _DOWNLOAD_LOCK:
        current_configuration = (
            _artifact_repo_id,
            _artifact_token,
            _artifact_revision,
            _artifact_local_dir,
        )
        if next_configuration != current_configuration:
            (
                _artifact_repo_id,
                _artifact_token,
                _artifact_revision,
                _artifact_local_dir,
            ) = next_configuration
            _configuration_version += 1
            _download_snapshot_cached.cache_clear()


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
def _download_snapshot_cached(
    repo_id: str,
    revision: str | None,
    token: str | None,
    required_files: tuple[str, ...],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub import errors as hub_errors
    except ImportError as exc:
        raise ArtifactStoreError(
            "External artifact storage is configured, but huggingface_hub is not installed. "
            "Install project dependencies from requirements.txt."
        ) from None

    expected_hub_errors = (
        "EntryNotFoundError",
        "FileMetadataError",
        "GatedRepoError",
        "HfHubHTTPError",
        "IncompleteSnapshotError",
        "LocalEntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "XetDownloadError",
    )
    hub_error_types = tuple(
        error_type
        for error_name in expected_hub_errors
        if isinstance((error_type := getattr(hub_errors, error_name, None)), type)
    )
    handled_error_types = hub_error_types + (OSError, TypeError, ValueError)
    def available_error_types(*names: str) -> tuple[type[BaseException], ...]:
        return tuple(
            error_type
            for name in names
            if isinstance((error_type := getattr(hub_errors, name, None)), type)
        )

    entry_not_found = available_error_types("EntryNotFoundError")
    gated_repo = available_error_types("GatedRepoError")
    repository_not_found = available_error_types(
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    )

    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            allow_patterns=[f"processed/{name}" for name in required_files],
        )
    except handled_error_types as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403} or isinstance(exc, gated_repo):
            reason = (
                "Hugging Face denied access. Verify the private dataset repository ID "
                "and that the configured read token has access."
            )
        elif isinstance(exc, entry_not_found):
            reason = (
                "The requested files or revision were not found. Verify that every file "
                "exists under the exact processed/ paths listed below."
            )
        elif isinstance(exc, repository_not_found):
            reason = (
                "The dataset repository or revision was not found or is not visible to "
                "the configured token. Verify the owner/repository ID and revision."
            )
        else:
            reason = (
                "The Hub download failed. Check the repository, token permissions, "
                "revision, network availability, and available disk space."
            )
        expected_paths = ", ".join(f"processed/{name}" for name in required_files)
        raise ArtifactStoreError(
            f"Could not download GraphSentinel artifacts from dataset '{repo_id}' "
            f"(revision '{revision or 'default'}'). {reason} Expected remote paths: "
            f"{expected_paths}. Failure type: {type(exc).__name__}."
        ) from None
    return Path(snapshot).resolve() / "processed"


def _download_snapshot(
    repo_id: str,
    revision: str | None,
    token: str | None,
    required_files: tuple[str, ...],
) -> Path:
    with _DOWNLOAD_LOCK:
        return _download_snapshot_cached(repo_id, revision, token, required_files)


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
        missing = [
            f"processed/{name}"
            for name in required
            if not (snapshot_dir / name).is_file()
        ]
        if not missing:
            empty = [
                path
                for path in required
                if (snapshot_dir / path).stat().st_size == 0
            ]
            if empty:
                raise ArtifactStoreError(
                    "Downloaded artifacts are empty at: "
                    + ", ".join(f"processed/{name}" for name in empty)
                    + f" in dataset '{repo_id}'. Re-upload complete files."
                )
            return snapshot_dir
        raise ArtifactStoreError(
            f"Dataset '{repo_id}' is missing required remote paths: "
            + ", ".join(missing)
            + ". Upload each file at exactly the listed path (do not nest another "
            "processed/ directory)."
        )

    missing = [name for name in required if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required GraphSentinel files are missing from {local_dir}: "
            + ", ".join(missing)
            + ". For Streamlit Cloud, set [artifacts].repo_id and [artifacts].token "
            "in App settings → Secrets. Expected remote paths: "
            + ", ".join(f"processed/{name}" for name in missing)
            + "."
        )
    empty = [name for name in required if (local_dir / name).stat().st_size == 0]
    if empty:
        raise ArtifactStoreError(
            f"Required GraphSentinel files are empty in {local_dir}: "
            + ", ".join(empty)
        )
    return local_dir


def configure_from_mapping(secrets: Mapping[str, object]) -> None:
    """Read the optional [artifacts] Streamlit Secrets table."""
    values = secrets.get("artifacts", {})
    if not isinstance(values, Mapping):
        raise ValueError("Streamlit Secrets [artifacts] must be a table.")
    repo_id = values.get("repo_id")
    token = values.get("token")
    if repo_id and not token:
        raise ValueError(
            "Streamlit Secrets [artifacts].token is missing. Add a read-only token "
            "with access to the private Hugging Face dataset."
        )
    configure_artifact_store(
        repo_id=str(repo_id) if repo_id else None,
        token=str(token) if token else None,
        revision=str(values["revision"]) if values.get("revision") else None,
    )
