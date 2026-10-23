from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PUBLIC_DATASET_REPO_ID = "BSaiCharan-HF/graphsentinel-data"
PUBLIC_DATASET_REVISION = "main"
_artifact_local_dir: str | None = None
_configuration_version = 0
_DOWNLOAD_LOCK = threading.Lock()


class ArtifactStoreError(FileNotFoundError):
    """Raised when configured external artifacts cannot be resolved."""


def configure_artifact_store() -> None:
    global _artifact_local_dir, _configuration_version
    local_dir = _configured_local_dir()
    next_local_dir = str(local_dir) if local_dir else None
    with _DOWNLOAD_LOCK:
        if next_local_dir != _artifact_local_dir:
            _artifact_local_dir = next_local_dir
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
    network_error_types: list[type[BaseException]] = []
    try:
        from httpx2 import RequestError

        network_error_types.append(RequestError)
    except ImportError:
        pass
    try:
        from requests.exceptions import RequestException

        network_error_types.append(RequestException)
    except ImportError:
        pass
    handled_error_types = (
        hub_error_types
        + tuple(network_error_types)
        + (OSError, TypeError, ValueError)
    )

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
            token=False,
            allow_patterns=[f"processed/{name}" for name in required_files],
        )
    except handled_error_types as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403} or isinstance(exc, gated_repo):
            reason = (
                "Hugging Face denied anonymous access. The dataset must be public and "
                "ungated for token-free downloads."
            )
        elif isinstance(exc, entry_not_found):
            reason = (
                "The requested files or revision were not found. Verify that every file "
                "exists under the exact processed/ paths listed below."
            )
        elif isinstance(exc, repository_not_found):
            reason = (
                "The public dataset repository or revision was not found. Verify the "
                "configured repository ID and revision."
            )
        else:
            reason = (
                "The anonymous Hub download failed. Check that the repository is public, "
                "the revision and remote paths are correct, and disk/network access is available."
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
    required_files: tuple[str, ...],
) -> Path:
    with _DOWNLOAD_LOCK:
        return _download_snapshot_cached(repo_id, revision, required_files)


def resolve_processed_dir(required_files: Iterable[str] = ()) -> Path:
    required = tuple(sorted(set(required_files)))
    configured_dir = _configured_local_dir()
    local_dir = configured_dir or LOCAL_PROCESSED_DIR
    if all((local_dir / name).is_file() for name in required):
        return local_dir

    if required:
        snapshot_dir = _download_snapshot(
            PUBLIC_DATASET_REPO_ID,
            PUBLIC_DATASET_REVISION,
            required,
        )
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
                    + f" in public dataset '{PUBLIC_DATASET_REPO_ID}'."
                )
            return snapshot_dir
        raise ArtifactStoreError(
            f"Public dataset '{PUBLIC_DATASET_REPO_ID}' is missing required remote paths: "
            + ", ".join(missing)
            + ". Upload each file at exactly the listed path (do not nest another "
            "processed/ directory)."
        )

    missing = [name for name in required if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required GraphSentinel files are missing from {local_dir}: "
            + ", ".join(missing)
            + f". The app downloads missing files anonymously from public dataset "
            f"'{PUBLIC_DATASET_REPO_ID}'. Expected remote paths: "
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
