from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.fraud_model import train_baseline_model


if __name__ == "__main__":
    artifact = train_baseline_model(save_artifacts=True)
    print(f"Baseline model trained. Threshold: {artifact['selected_threshold']:.4f}")
    print(f"Validation F1: {artifact['validation_summary']['f1']:.4f}")
    print(f"Test PR-AUC: {artifact['test_summary']['pr_auc']:.4f}")
