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

## Local development setup (maintainer)

Visitors to a deployed public demo only need its published Streamlit URL. The public dataset repository is configured in the application; reviewers do not upload files, provide credentials, or run commands.

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

- The public demo does not ask visitors to upload files, provide credentials, or run commands.
- On application startup, the app retrieves the required processed splits and matching model artifacts from configured private storage, validates them, and prepares transaction predictions before rendering the data-dependent pages.
- Downloaded artifacts use the Hugging Face Hub cache; loaded predictions and the model are cached across Streamlit reruns. The app does not retrain.
- If artifact storage is unavailable or artifacts fail validation, the app records the error in the Streamlit application logs and shows a user-safe fallback. Transaction pages do not show fabricated or empty placeholder results; Methodology & Data remains available.
- Training and evaluation remain reproducible because the selected validation threshold and held-out test metrics are saved to data/processed/evaluation_summary.json.
- The operating threshold is selected by maximizing validation-set F1; when multiple thresholds tie, the higher threshold is preferred. The held-out test set is used only for final metric reporting.
- Model Lab reports confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, ROC/PR curves, and threshold trade-offs when required outputs are present. Threshold analysis uses validation predictions; final test metrics use the held-out test split.
- The dashboard includes the Overview, Transaction Explorer, Graph Intelligence, Fraud Investigation, Model Lab, and Methodology & Data pages.

## Streamlit Community Cloud deployment

The raw dataset and generated split CSVs are intentionally excluded from Git. The data-facing pages need `train.csv`, `validation.csv`, and `test.csv` at runtime; inference and evaluation also require a matching `fraud_model.joblib` and `evaluation_summary.json`. These are developer-only deployment prerequisites; visitors do not configure or upload them. The Methodology & Data page remains available if deployment data is unavailable.

This app downloads anonymously from the public Hugging Face Hub dataset `BSaiCharan-HF/graphsentinel-data` on revision `main`. The repository must be public and ungated. Place the five files at the repository root:

```text
train.csv
validation.csv
test.csv
fraud_model.joblib
evaluation_summary.json
```

The downloader stores/reuses the Hub snapshot in its cache and returns that snapshot directory to the existing loaders, which expect these filenames directly inside the resolved processed-artifact directory. Do not upload `data/raw/creditcard.csv`. After artifact updates, run `python src/models/train_baseline.py` locally so the summary contains the SHA-256 checksum of its matching model file, then upload the model and summary together. Never edit the checksum by hand.

No Hugging Face token, Streamlit Secret, repository-ID input, or environment variable is required. The app calls the Hub snapshot downloader with `repo_type="dataset"` and explicitly disables authentication. It requests only the five root-level files, downloads large files into the Hugging Face cache, and reuses the cached snapshot and parsed model/predictions across reruns. It does not download the raw dataset or retrain. The split CSVs total roughly 155 MB, so the host needs adequate network transfer, disk cache, and memory to load and score them.

Before launch, the maintainer must ensure the Hugging Face repository is genuinely public and that the five files are present at the exact root-level paths shown above. This source configuration is fixed in `src/data/artifact_store.py`; no app settings are needed. Reboot the Streamlit app after updating the dataset. If startup fails, application logs show the repository, failed remote path(s), and a safe error category. A 401/403 means Hugging Face is not allowing anonymous access; a missing-file diagnostic lists the expected root-level filenames.

For local development only, complete local `data/processed/` files are used before downloading. `GRAPHSENTINEL_PROCESSED_DIR` can point to an alternate local processed-data directory; it is not required for the deployed demo.

After pushing application changes, use **Manage app → Reboot** (or wait for repository sync/redeploy) in Streamlit Community Cloud. Verify that Overview displays transaction statistics, Transaction Explorer displays records, Graph Intelligence renders inferred feature-similarity edges, Fraud Investigation loads cases and signed explanations, and Model Lab displays split-based metrics. Local tests do not verify that the public Hugging Face repository is accessible or that the hosted deployment succeeds.

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
