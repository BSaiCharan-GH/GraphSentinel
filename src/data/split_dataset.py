import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "split_summary.json"
SEED = 42


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    df = pd.read_csv(SOURCE)

    if "Class" not in df.columns:
        raise ValueError("Required target column 'Class' is missing.")

    if set(df["Class"].dropna().unique()) != {0, 1}:
        raise ValueError("Expected Class labels 0 (legitimate) and 1 (fraud).")

    if df.isna().any().any():
        raise ValueError("Missing values detected. Resolve them before splitting.")

    # 70% train, 15% validation, 15% test.
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["Class"],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_df["Class"],
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }

    summary = {
        "random_seed": SEED,
        "split_strategy": "stratified random split",
        "split_sizes": {},
        "warning": (
            "Random stratification is a baseline protocol. "
            "Time-based evaluation may be needed to assess deployment realism."
        ),
    }

    for filename, part in splits.items():
        path = OUTPUT / filename
        part.to_csv(path, index=False)

        fraud_count = int((part["Class"] == 1).sum())
        summary["split_sizes"][filename] = {
            "rows": int(len(part)),
            "fraud_cases": fraud_count,
            "legitimate_cases": int((part["Class"] == 0).sum()),
            "fraud_rate_percent": round(100 * fraud_count / len(part), 6),
            "path": str(path.relative_to(ROOT)),
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== SPLIT SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nSaved train, validation, and test datasets.")


if __name__ == "__main__":
    main()
