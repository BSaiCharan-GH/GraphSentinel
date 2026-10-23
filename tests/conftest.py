import sys
from pathlib import Path

import pytest

from src.data.artifact_store import LOCAL_PROCESSED_DIR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def processed_splits() -> None:
    required = [
        LOCAL_PROCESSED_DIR / "train.csv",
        LOCAL_PROCESSED_DIR / "validation.csv",
        LOCAL_PROCESSED_DIR / "test.csv",
    ]
    missing = [
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        for path in required
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            "Integration test requires processed split files: "
            + ", ".join(missing)
            + ". Generate them with: python src/data/validate_dataset.py, "
            "then python src/data/split_dataset.py."
        )


@pytest.fixture
def trained_artifacts(processed_splits: None) -> None:
    required = [
        LOCAL_PROCESSED_DIR / "fraud_model.joblib",
        LOCAL_PROCESSED_DIR / "evaluation_summary.json",
    ]
    missing = [
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        for path in required
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            "Integration test requires trained model artifacts: "
            + ", ".join(missing)
            + ". First generate the splits with python src/data/split_dataset.py, "
            "then run python src/models/train_baseline.py."
        )
