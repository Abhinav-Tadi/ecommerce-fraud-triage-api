# Design Decisions Log

---

## Problem & Dataset

- **Problem:** Predict whether a card-not-present e-commerce transaction is fraudulent,
  framed as flagging high-risk transactions for manual review rather than emitting
  a raw probability score.

- **Dataset:** IEEE-CIS Fraud Detection (Kaggle competition, 2019), provided by Vesta
  Corporation via the IEEE Computational Intelligence Society.
  Source: kaggle.com/c/ieee-fraud-detection
  ~590,540 labeled transactions, 431 raw features (400 numerical, 31 categorical),
  ~3.5% fraud rate, two tables requiring a join (train_transaction + train_identity;
  identity coverage is partial — not every transaction has a matching identity record).

- **Why this dataset over ULB credit-card-fraud or PaySim:**
  ULB's data is already PCA-transformed, which removes any real feature-engineering
  work and makes a business-decision-style demo impossible (inputs are anonymous
  components, not interpretable fields). PaySim is synthetic — simulator output, not
  real transactions — which fails the real-world data requirement outright.
  IEEE-CIS is real, genuinely messy (hundreds of sparse/missing columns, a two-table
  join with partial coverage), and includes human-readable fields (amount, product
  code, card network, device type, email domain) that support a genuine
  business-decision-framed demo.

- **Why real-time, not batch:** A fraud/no-fraud decision is needed at the moment a
  transaction is authorised. There is no requirement to score transactions in bulk
  after the fact, and the model and feature set are small enough to serve
  synchronously with acceptable latency.

---

## Evaluation Metric
*Logged before model training.*

- **Primary metric:** PR-AUC (Precision-Recall Area Under Curve)
- **Secondary metric:** F1 score on the fraud class at the chosen operating threshold

- **Why not accuracy:** At 3.5% fraud rate, predicting "not fraud" for every
  transaction scores 96.5% accuracy. Accuracy is a degenerate metric for this
  problem — it rewards ignoring the task entirely.

- **Why PR-AUC over ROC-AUC:** ROC-AUC is optimistic under class imbalance because
  it accounts for true negatives, which are abundant and easy to get right. PR-AUC
  focuses on performance on the minority class and is more informative when false
  positives carry real operational cost (manual review team capacity).

- **Why PR-AUC over a single F1 score:** F1 at a fixed threshold is sensitive to
  which threshold is chosen. PR-AUC evaluates the model across all possible thresholds
  and gives a complete picture before the final operating threshold is selected.

- **Threshold:** Not 0.5 by default. Will be chosen after training to reflect the
  asymmetric cost of a missed fraud (high) versus a false positive (review burden).
  The specific threshold and its business rationale will be documented here once set.

---

## EDA Findings

- **Confirmed fraud rate:** 3.499% (20,663 fraud / 569,877 non-fraud out of 590,540)
- **Identity coverage:** 24.4% of transactions have identity records.
  75.6% have no identity data whatsoever. This is the dominant case, not an edge case.
- **Total columns after join:** 434
- **Columns with any missing values:** 414
- **Columns with >50% missing:** 214
- **Columns with >80% missing:** 74

- **TransactionAmt:** Heavily right-skewed (mean 135, median 69, max 31,937).
  log1p transformation applied.

- **Top features by importance (leakage audit):** C8, V94, V34, V70, V317, V91,
  V308, C14. No TransactionID, time index, or post-outcome field appeared in the
  top 20. Leakage audit passed — proceeding with training.

---

## Join Strategy (train_transaction + train_identity)

- **Approach:** Left join on TransactionID — every transaction gets a row; identity
  fields are NaN where no matching identity record exists.

- **Why not an inner join:** Dropping transactions without identity records would bias
  the training set. We do not know whether identity-absent transactions have a
  systematically different fraud rate — discarding them would be assuming the answer.

- **Inference-time implication:** The API must accept requests where identity fields
  are absent and treat them as NaN, exactly as they appear in training. Missing
  identity features are not errors — they are the majority input state (75.6% of
  all transactions). The preprocessing function must handle this explicitly and
  must not error on absent identity fields.

---

## Missing Value Strategy

- **Decision:** Pass NaN directly to XGBoost. No imputation. No sentinel filling
  (e.g. -999). No column dropping based on missingness rate.

- **Why:** XGBoost handles NaN natively — during training it learns the optimal
  direction to send missing values at each split. Filling with -999 treats missingness
  as a real numeric value and corrupts this learned behaviour. Dropping high-missingness
  columns preemptively discards signal: the 74 columns with >80% missing are
  product-specific (V-blocks) or identity-specific (id_ columns) — their missingness
  pattern is structurally informative, not random noise.

- **Exception:** The logistic regression baseline cannot handle NaN. Median imputation
  was applied for the baseline only, inside a pipeline that is not reused at inference
  time. The final XGBoost model receives NaN directly.

---

## Class Imbalance Strategy

- **Confirmed class ratio:** 569,877 non-fraud / 20,663 fraud = 27.6:1
- **Decision:** `scale_pos_weight=27.6` in XGBoost, computed directly from training
  data as (negative count / positive count).
- **Why not SMOTE:** SMOTE adds complexity and can distort the feature space on
  high-dimensional tabular data. Class weighting achieves the same correction to the
  loss function without synthetic data generation.
- **Threshold tuning:** Will be applied post-training to reflect the asymmetric
  business cost of false negatives vs false positives. The operating threshold will
  not default to 0.5.

---

## TransactionAmt Transformation

- **Decision:** log1p transform applied at both training time and inference time.
- **Why:** Raw distribution is heavily right-skewed (mean 135, max 31,937).
  log1p chosen over log because it handles zero-value transactions cleanly (log(0)
  is undefined; log1p(0) = 0).
- **Critical:** This transformation must be applied in `scripts/preprocess.py` and
  called identically at inference time. A transformation applied at training time
  but not at inference time is silent leakage in reverse.

---

## Categorical Encoding

- **Decision:** pandas category codes (`.astype('category').cat.codes`), with -1
  (the code for NaN) replaced by NaN so XGBoost handles missing categoricals natively.
- **Columns encoded:** All object/string dtype columns — ProductCD, card4, card6,
  P_emaildomain, R_emaildomain, and id_ string columns.
- **Why not one-hot encoding:** With high-cardinality categoricals (email domains
  have hundreds of values), one-hot encoding would explode dimensionality. XGBoost
  handles integer-encoded categoricals natively and learns non-linear splits on them.

---

## Model Selection

- **Baseline (Logistic Regression):**
  - PR-AUC: 0.4388
  - F1 on fraud class: 0.23 at threshold 0.5
  - Note: baseline used median imputation and standard scaling — neither of which
    is used in the final model.

- **XGBoost (current result — interim, model not yet converged):**
  - PR-AUC: 0.6973
  - Parameters: n_estimators=500, max_depth=6, learning_rate=0.05,
    scale_pos_weight=27.6, subsample=0.8, colsample_bytree=0.8
  - Early stopping: DID NOT TRIGGER — model hit the n_estimators=500 hard limit
    at iteration 499, meaning it was still improving. 0.6973 is a lower bound.
  - Next step: re-run with n_estimators=1000, record the iteration where early
    stopping fires, and update this section with the converged PR-AUC.

- **Improvement over baseline:** +0.2585 PR-AUC (interim)

- **Model size:** [TO BE FILLED — run joblib.dump and check file size before Phase 2]

- **What the model gets wrong:** [TO BE FILLED after final evaluation — required
  before this project is called done]

- **Why XGBoost over LightGBM:** Not directly compared. XGBoost was chosen for
  its native NaN handling, mature Lambda deployment track record, and sufficient
  performance for this use case. LightGBM would be the first thing to try if
  training speed or model size became a constraint.

- **Why classical ML over deep learning:** Keeps the model artifact in the
  tens-of-MB range, avoids Lambda cold-start pain from loading large models, and
  requires explicit feature engineering decisions rather than delegating
  representation learning to a network.

- **Hyperparameter tuning:** RandomizedSearchCV — to be run once the model
  converges with early stopping. Not full GridSearchCV; this is not a Kaggle
  leaderboard optimisation.

---

## Architecture

- **Inference:** AWS Lambda (container image via ECR) + API Gateway HTTP API
- **Model storage:** S3 (artifact versioning, separate from the container)
- **Logging:** DynamoDB (prediction log, Phase 7) + CloudWatch (Lambda logs)
- **Frontend:** Streamlit, hosted on Streamlit Community Cloud (not AWS)

- **Why Lambda + API Gateway over EC2:**
  AWS accounts created post-mid-2025 use a credit-based free tier (~$100–200,
  expires after six months or when exhausted). EC2 draws down that credit balance
  and generates charges when left running. Lambda and API Gateway sit on AWS's
  permanent Always Free allowances (1M Lambda requests/month, 1M API Gateway
  calls/month) and run indefinitely at near-zero cost after credits expire. This
  keeps the project live for the portfolio long-term without ongoing cost.

- **Why not SageMaker:** Same credit-pool problem; SageMaker's own free-tier
  allowances were designed for the old 12-month free model and are unreliable on
  new accounts.

- **Known simplification — cold starts:** Lambda adds latency on the first request
  after an idle period. Acceptable for a portfolio demo; provisioned concurrency
  would eliminate this in production.

- **Why the Streamlit frontend is hosted separately:**
  Removes all AWS cost risk from the UI layer. Streamlit Community Cloud is free
  and requires no AWS configuration. This is a deliberate infrastructure decision,
  not a shortcut.

---

## IAM & AWS Setup

- **IAM user:** abhinavtadi-dev
- **Policies attached:** AmazonEC2ContainerRegistryFullAccess, AWSLambda_FullAccess,
  AmazonAPIGatewayAdministrator, AmazonS3FullAccess, CloudWatchLogsFullAccess
- **Known simplification:** Broader than least-privilege. In production, permissions
  would be scoped to exactly what each service requires.
- **Root account:** MFA enabled, no access keys generated, not used for any
  resource creation.
- **AWS Region:** us-east-1. Originally configured as ap-south-2 (Hyderabad), which
  is an opt-in region requiring explicit account activation — STS calls returned
  InvalidClientTokenId even with valid credentials. Switched to us-east-1.
- **Billing alert:** $1.00 monthly cost budget, 100% actual-spend alert, email
  notification configured before any resource was created.

---

## Monitoring

- **Implemented:** DynamoDB prediction logging (Phase 7 — planned, not yet built).
  Each prediction will log: request UUID, input features, prediction, probability
  score, timestamp.

- **Not implemented:** Automated drift detection.

- **What I would build next:** A scheduled Lambda (EventBridge trigger, weekly) that
  queries the last 7 days of DynamoDB prediction logs, computes the distribution of
  key input features (TransactionAmt, card type, device type), and compares against
  the training distribution using a KS test or Population Stability Index (PSI > 0.2
  threshold). Alerts via SNS on significant shift or if predicted fraud rate moves
  more than 2 standard deviations from the training baseline.

- **Why not built for v1:** Outside the scope of the current project timeline.
  Manual monitoring via CloudWatch and periodic DynamoDB inspection is sufficient
  to demonstrate the concept. The design above is what would be implemented given
  another two to three weeks.