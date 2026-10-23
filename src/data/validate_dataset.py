import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_PATH = ROOT / "reports" / "dataset_validation.json"

EXPECTED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download creditcard.csv and place it in data/raw/."
        )

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in EXPECTED_COLUMNS if col not in df.columns
    ]
    extra_columns = [
        col for col in df.columns if col not in EXPECTED_COLUMNS
    ]

    numeric_columns = [
        col for col in EXPECTED_COLUMNS if col in df.columns
    ]
    non_numeric = []
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    report = {
        "dataset": DATA_PATH.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_expected_columns": missing_columns,
        "unexpected_columns": extra_columns,
        "non_numeric_columns": non_numeric,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "validation_passed": False,
        "checks": {},
    }

    if "Class" in df.columns:
        labels = set(df["Class"].dropna().unique().tolist())
        report["class_distribution"] = {
            str(label): int(count)
            for label, count in df["Class"].value_counts().items()
        }
        fraud_count = int((df["Class"] == 1).sum())
        report["fraud_count"] = fraud_count
        report["fraud_percentage"] = (
            round(100 * fraud_count / len(df), 6) if len(df) else 0
        )
        report["checks"]["binary_class_labels"] = labels.issubset({0, 1})
    else:
        report["checks"]["binary_class_labels"] = False

    if all(col in df.columns for col in ["Time", "Amount"]):
        report["checks"]["nonnegative_time"] = bool(
            pd.to_numeric(df["Time"], errors="coerce").ge(0).all()
        )
        amount = pd.to_numeric(df["Amount"], errors="coerce")
        report["checks"]["nonnegative_amount"] = bool(amount.ge(0).all())
    else:
        report["checks"]["nonnegative_time"] = False
        report["checks"]["nonnegative_amount"] = False

    if not non_numeric:
        numeric = df[numeric_columns].to_numpy(dtype=np.float64)
        report["checks"]["finite_numeric_values"] = bool(
            np.isfinite(numeric).all()
        )
    else:
        report["checks"]["finite_numeric_values"] = False

    report["checks"]["expected_columns_present"] = not missing_columns
    report["checks"]["no_missing_values"] = missing_values == 0

    report["validation_passed"] = all(report["checks"].values())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n===== DATASET VALIDATION =====")
    print(f"Rows: {report['rows']:,}")
    print(f"Columns: {report['columns']}")
    print(f"Fraud cases: {report.get('fraud_count', 'Unavailable')}")
    print(f"Fraud percentage: {report.get('fraud_percentage', 'Unavailable')}%")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Missing values: {missing_values:,}")

    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    print(f"\nReport saved to: {REPORT_PATH}")
    print("OVERALL:", "PASS" if report["validation_passed"] else "FAIL")

    if not report["validation_passed"]:
        raise SystemExit(
            "Dataset validation failed. Inspect the report before proceeding."
        )


if __name__ == "__main__":
    main()
