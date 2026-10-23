# GraphSentinel for Razorpay

GraphSentinel is an explainable fraud-intelligence console for payment-risk analysis. The project uses the public ULB credit-card fraud dataset, builds a tabular fraud-detection baseline, and surfaces model confidence, graph-based similarity relationships, and investigation workflows in a Streamlit dashboard.

## Product goals

- Validate the public dataset and produce reproducible splits.
- Train and evaluate a fraud baseline without fabricating real payment activity.
- Present explainable alerts and transaction-level evidence.
- Construct a defensible similarity graph from transaction features only.
- Deliver a polished, portfolio-ready dashboard for analysis and review.

## Dataset

ULB / Worldline Credit Card Fraud Detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Expected file: data/raw/creditcard.csv

Expected columns:
- Time
- V1 through V28
- Amount
- Class

The dataset contains anonymized PCA-transformed features. It does not include real customers, merchants, cards, or payment network edges, so any graph is explicitly treated as an inferred analytical relationship rather than a real-world payment graph.

## Setup

1. Create and activate the project virtual environment:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
2. Install dependencies:
   python -m pip install -r requirements.txt
3. Validate the raw dataset:
   python src/data/validate_dataset.py
4. Create train/validation/test splits:
   python src/data/split_dataset.py
5. Train the fraud baseline:
   python src/models/train_baseline.py
6. Run the test suite:
   python -m pytest -q
7. Launch the dashboard:
   streamlit run app.py

For a clean checkout, use the commands in that order after placing `creditcard.csv` at `data/raw/creditcard.csv`. Tests that need processed splits or trained artifacts report a clear pytest skip when those prerequisites are absent; standalone signed-explanation, threshold-selection, and synthetic graph-leakage tests do not require the dataset or saved model.

## Expected runtime behavior

- If data/raw/creditcard.csv is missing, the app shows a friendly setup message instead of crashing.
- If the processed model artifacts are absent, the app prompts the user to train the baseline before exploring predictions.
- Training and evaluation remain reproducible because the selected validation threshold and held-out test metrics are saved to data/processed/evaluation_summary.json.
- The operating threshold is selected by maximizing validation-set F1; when multiple thresholds tie, the higher threshold is preferred. The held-out test set is used only for final metric reporting.
- Opening the dashboard does not train a model implicitly. Missing model artifacts display setup guidance; run the documented training command to create them.
- Model Lab reports confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, ROC/PR curves, and threshold trade-offs when required outputs are present. Threshold analysis uses validation predictions; final test metrics use the held-out test split.
- The dashboard includes the Overview, Transaction Explorer, Graph Intelligence, Fraud Investigation, Model Lab, and Methodology & Data pages.

## Repository structure

- app.py — Streamlit dashboard entry point
- src/data/validate_dataset.py — dataset quality validation
- src/data/split_dataset.py — stratified split generation
- src/models/fraud_model.py — baseline model logic and evaluation
- src/models/train_baseline.py — model training command
- src/graph/graph_builder.py — similarity graph construction
- reports/ — validation and split summaries
- data/processed/ — generated train/validation/test CSVs and model artifacts

## Notes

- This project does not connect to Razorpay live infrastructure.
- No customer or merchant identities are inferred from the dataset.
- Model scores should be interpreted as risk signals for analysis, not as proof of fraud.
