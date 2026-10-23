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

Visitors to a deployed public demo only need its published Streamlit URL. Dataset access and secrets are configured once by the app owner; reviewers do not upload files, provide credentials, or run commands.

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

This app uses a private Hugging Face Hub **dataset repository** as external artifact storage. Upload only these files from `data/processed/`, preserving the `processed/` folder:

```text
processed/train.csv
processed/validation.csv
processed/test.csv
processed/fraud_model.joblib
processed/evaluation_summary.json
```

Do not upload `data/raw/creditcard.csv`. After artifact updates, run `python src/models/train_baseline.py` locally so the summary contains the SHA-256 checksum of its matching model file, then upload the model and summary together. Never edit the checksum by hand.

As the app owner, configure the deployed Streamlit app once under **App settings → Secrets**:

```toml
[artifacts]
repo_id = "YOUR_HUGGING_FACE_ACCOUNT/graphsentinel-artifacts"
token = "YOUR_READ_ONLY_HUGGING_FACE_TOKEN"
revision = "main"
```

`repo_id` should normally be `account/repository`; a dataset URL such as `https://huggingface.co/datasets/account/repository` is also accepted. Create a private dataset repository and a read-only token with access to it. Keep the token only in Streamlit Secrets; it is never shown in the application or committed to Git. The app downloads and validates the complete required artifact bundle on first startup, then reuses the Hugging Face Hub disk cache and Streamlit's in-process caches across reruns. It does not download the raw dataset or retrain. The three split CSVs total roughly 155 MB, so the host needs enough disk, network transfer, and memory to load and score them.

After setting the secrets and uploading the artifact bundle, reboot the app once and review its logs. Successful first-load validation is required before the public demo is ready. If startup fails, the technical cause is written to application logs while visitors see a safe fallback instead of invented results.

For local use, local `data/processed/` files are preferred. Alternatively set `GRAPHSENTINEL_PROCESSED_DIR` to a directory containing the five files above. `GRAPHSENTINEL_ARTIFACT_REPO`, `GRAPHSENTINEL_HF_TOKEN`, and `GRAPHSENTINEL_ARTIFACT_REVISION` may be used instead of Streamlit Secrets for non-Cloud environments.

After pushing application changes, use **Manage app → Reboot** (or wait for repository sync/redeploy) in Streamlit Community Cloud. As the app owner, verify in the hosted app that Overview displays transaction statistics, Transaction Explorer displays records, Graph Intelligence renders inferred feature-similarity edges, Fraud Investigation loads cases and signed explanations, and Model Lab displays split-based metrics. If initialization fails, inspect app logs and verify the repository ID, read permission, revision, file paths, model checksum, and available memory. A successful local test does not itself verify the hosted deployment.

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
