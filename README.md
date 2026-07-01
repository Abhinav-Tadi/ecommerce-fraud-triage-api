# ecommerce-fraud-triage-api

> **Status: Phases 1–3 complete (model training, API packaging, AWS account setup). Phase 4 — cloud deployment — is next. Live endpoint and demo will be linked here once it's running.**

---

## What This Is

Real-time fraud triage for card-not-present e-commerce transactions. Send a transaction's features, get back a binary flag (review or pass) and a probability score.

Built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) from Vesta Corporation — ~590K real transactions, 3.5% fraud rate, two joined tables with genuinely messy features.

The threshold isn't 0.5. It's set at 0.0957 to hit 85% recall, which reflects what a missed fraud actually costs versus the cost of a false-positive review. The reasoning is in [DECISIONS.md](./DECISIONS.md).

---

## Live Demo

Not live yet — link will appear here once Phase 4 is done.

---

## Architecture

```mermaid
flowchart LR
    A["Streamlit Demo\n(Streamlit Community Cloud)"] -->|HTTPS POST| B["API Gateway\nHTTP API"]
    B --> C["Lambda\nDocker container via ECR"]
    C --> D["XGBoost model\nmodel.ubj"]
    C --> E["CloudWatch Logs"]
    C --> F["DynamoDB\nprediction log"]
    G["S3 bucket"] -.->|model artifact versioning| C
```

Inference runs on AWS. The Streamlit frontend runs on Streamlit Community Cloud — keeping it off AWS entirely removes any billing risk from the UI layer.

---

## Build Status

| Component | Status | Notes |
|---|---|---|
| Problem & dataset selection | ✅ Done | IEEE-CIS Fraud Detection; see DECISIONS.md |
| EDA & feature engineering | ✅ Done | notebooks/01_eda.ipynb |
| Leakage audit | ✅ Done | Run during EDA; TransactionID and time-index confirmed clean |
| Baseline model (logistic regression) | ✅ Done | PR-AUC 0.4393 |
| XGBoost classifier | ✅ Done | PR-AUC 0.8691, 422 features, threshold 0.0957 |
| scripts/preprocess.py | ✅ Done | Same code path used at training time and inference time |
| FastAPI inference endpoint | ✅ Done | /predict + /health; 4/4 curl tests passing locally |
| Docker containerisation | ✅ Done | Python 3.12 base image; tested with Lambda v2 event format locally |
| AWS billing alert (Phase 3) | ✅ Done | $1 budget alert configured before any resource was created |
| IAM user — abhinavtadi-dev (Phase 3) | ✅ Done | Scoped policies; root account not used for anything |
| ECR image push | 🔲 Phase 4 | |
| Lambda function | 🔲 Phase 4 | Must use --architectures arm64 — see DECISIONS.md |
| API Gateway HTTP API | 🔲 Phase 4 | |
| S3 model artifact storage | 🔲 Phase 4 | |
| Streamlit demo | 🔲 Phase 5 | Hosting on Streamlit Community Cloud, not AWS |
| DynamoDB prediction logging | 🔲 Phase 7 | |
| GitHub Actions CI/CD | 🔲 Stretch goal | Redeploys Lambda on push to main |
| Automated drift detection | ❌ Not built | Design is documented in DECISIONS.md |

---

## Key Design Decisions

Full reasoning in [DECISIONS.md](./DECISIONS.md). Short version:

**Dataset — IEEE-CIS over ULB or PaySim**
ULB is already PCA-transformed. The inputs are anonymous components, not transaction fields, which makes a real inference API pointless — you can't build a business-framed demo when you can't explain what the features mean. PaySim is synthetic. IEEE-CIS has real transactions with interpretable fields and a genuinely annoying join problem (75.6% of transactions have no matching identity record at all). That messiness is what makes it worth working through.

**Evaluation metric — PR-AUC, not accuracy**
At 3.5% fraud rate, predicting "not fraud" every single time gets you 96.5% accuracy. Precision and recall on the fraud class are the only numbers that say anything useful here.

**Lambda over EC2**
AWS accounts created after mid-2025 get a credit balance that expires after six months. EC2 draws it down. Lambda, API Gateway, DynamoDB, and S3 all sit on AWS's permanent Always Free tier. The project stays live indefinitely.

**XGBoost over deep learning**
41MB artifact, native NaN handling, no GPU needed. A deep learning model would have tripled cold-start latency and required GPU infrastructure for no meaningful accuracy gain on a tabular dataset this size.

---

## Run Locally

```bash
git clone https://github.com/Abhinav-Tadi/ecommerce-fraud-triage-api.git
cd ecommerce-fraud-triage-api

python3 -m venv venv
source venv/bin/activate

pip install -r requirements-dev.txt

uvicorn app.main:app --reload --port 8000
```

Test it:

```bash
# Minimal valid request (TransactionAmt is the only required field)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150.0}'

# With card and product fields
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 500.0, "ProductCD": "C", "card4": "discover", "card6": "credit", "P_emaildomain": "anonymous.com"}'

# Health check
curl http://localhost:8000/health

# Should return 422 — TransactionAmt is missing
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"card4": "visa"}'
```

The dataset isn't in this repo — download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) and place in `data/` to re-run training.

---

## What I'd Do Next

- **Drift detection** — a scheduled Lambda comparing the last 7 days of logged inputs against the training distribution. Not built; the design is in DECISIONS.md.
- **Least-privilege IAM** — current policies are broader than they need to be. Fine for a portfolio project, would tighten before anything touched production.
- **Infrastructure as code** — the Phase 4 deployment is CLI-based. Terraform would make it reproducible and version-controlled.

---

## Tech Stack

Python · XGBoost · scikit-learn · FastAPI · Pydantic · Docker · AWS Lambda · ECR · API Gateway · S3 · DynamoDB · CloudWatch · Streamlit

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — Vesta Corporation via Kaggle, 2019. ~590,540 transactions, 3.5% fraud rate.