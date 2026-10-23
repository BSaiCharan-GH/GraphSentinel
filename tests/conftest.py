import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def processed_splits() -> None:
    required = [
        ROOT / "data" / "processed" / "train.csv",
        ROOT / "data" / "processed" / "validation.csv",
        ROOT / "data" / "processed" / "test.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
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
        ROOT / "data" / "processed" / "fraud_model.joblib",
        ROOT / "data" / "processed" / "evaluation_summary.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        pytest.skip(
            "Integration test requires trained model artifacts: "
            + ", ".join(missing)
            + ". First generate the splits with python src/data/split_dataset.py, "
            "then run python src/models/train_baseline.py."
        )
