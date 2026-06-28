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

---

## EDA Findings

- **Confirmed fraud rate:** 3.499% (20,663 fraud / 569,877 non-fraud out of 590,540)
- **Identity coverage:** 24.4% of transactions have identity records.
  75.6% have no identity data whatsoever. This is the dominant inference-time case,
  not an edge case. The API must treat missing identity fields as NaN, not as errors.
- **Total columns after join:** 434
- **Columns with any missing values:** 414
- **Columns with >50% missing:** 214
- **Columns with >80% missing:** 74

**D-column missingness as fraud signal:**
D6, D7, D12, D13, D14 are missing 25–27 percentage points less often in fraud
transactions than in legitimate ones (e.g. D12: 63% missing for fraud vs 90%
for legitimate). These columns encode time-delta features — time since previous
transaction, previous event. Their presence indicates a transaction history pattern
associated with fraud. Fraudsters generate transaction sequences; legitimate
one-off purchasers often have no prior history. This confirms high-missingness
columns must not be dropped — their missingness encodes structural signal.

**Time-of-day fraud pattern:**
Fraud rate peaks at 7am (10.6%) and 8am (9.3%), with a low at 1pm (2.3%).
This is consistent with card-not-present fraud patterns — attacks cluster when
cardholders are asleep and not monitoring accounts. hour_of_day and
day_of_week_proxy were engineered from TransactionDT and retained as features.

**TransactionAmt:** Heavily right-skewed (mean $135, median $69, max $31,937).
Fraud transactions have a higher median amount ($75) than legitimate ($69) and
a higher mean ($149 vs $135). log1p transform applied.

**Feature importance — final model top features:**
V258 (24.7%), V201 (8.6%), V70 (3.7%), V122 (2.3%), V294 (2.1%).
No TransactionID or time-index feature appeared in either the leakage audit
or final model importances. Leakage audit passed.

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
  all transactions). The preprocessing function must handle this explicitly.

---

## Missing Value Strategy

- **Decision:** Pass NaN directly to XGBoost. No imputation. No sentinel filling
  (e.g. -999). No column dropping based on missingness rate.

- **Why:** XGBoost handles NaN natively — during training it learns the optimal
  direction to send missing values at each split. Filling with -999 treats missingness
  as a real numeric value and corrupts this learned behaviour. Dropping high-missingness
  columns preemptively discards signal: the 74 columns with >80% missing are
  product-specific (V-blocks) or identity-specific (id_ columns) — their missingness
  pattern is structurally informative, not random noise. D-column missingness
  differences of 25+ percentage points between fraud and legitimate transactions
  confirm this empirically.

- **Exception:** The logistic regression baseline cannot handle NaN. Median imputation
  was applied for the baseline only, inside a pipeline that is not reused at inference
  time. The final XGBoost model receives NaN directly.

---

## Class Imbalance Strategy

- **Confirmed class ratio:** 569,877 non-fraud / 20,663 fraud = 27.58:1
- **Decision:** `scale_pos_weight=27.58` in XGBoost, computed from y_train after
  the train/test split (not from the full dataset).
- **Why not SMOTE:** SMOTE adds complexity and can distort the feature space on
  high-dimensional tabular data. Class weighting achieves the same correction to
  the loss function without synthetic data generation.
- **Threshold tuning:** Applied post-training based on business cost asymmetry.
  The operating threshold is not 0.5. See Threshold Decision section.

---

## TransactionAmt Transformation

- **Decision:** log1p transform applied at both training time and inference time.
- **Why:** Raw distribution is heavily right-skewed (mean $135, max $31,937).
  log1p chosen over log because it handles zero-value transactions cleanly.
- **Critical:** This transformation is applied in scripts/preprocess.py and called
  identically at inference time. A transformation applied at training time but not
  at inference time is silent leakage in reverse — predictions would be made on a
  different scale than the model was trained on.

---

## Categorical Encoding

- **Decision:** pandas category codes (.astype('category').cat.codes), with -1
  (the code for NaN) replaced by NaN so XGBoost handles missing categoricals natively.
- **Columns encoded:** All object/string dtype columns — ProductCD, card4, card6,
  P_emaildomain, R_emaildomain, and id_ string columns.
- **Why not one-hot encoding:** With high-cardinality categoricals (email domains
  have hundreds of values), one-hot encoding would explode dimensionality. XGBoost
  handles integer-encoded categoricals natively and learns non-linear splits on them.

---

## TransactionDT — Time Feature Engineering

- **Raw column:** TransactionDT is a time delta in seconds from an undisclosed
  reference point. Used raw, it is a monotonically increasing index — not a
  meaningful feature. Dropped from the feature set.

- **Engineered features retained:**
  - hour_of_day: (TransactionDT % 86400) / 3600 — time of day in hours (0–24)
  - day_of_week_proxy: (TransactionDT // 86400) % 7 — 7-day repeating cycle

- **Why these features carry signal:** Fraud rate at 7am is 10.6% vs 2.3% at 1pm —
  a 4.6x difference. Dropping TransactionDT without analysis (as done in v1 of this
  notebook) was an error that left time-based signal on the table.

---

## Model Selection

- **Baseline (Logistic Regression):**
  - PR-AUC: 0.4393
  - Implementation: class_weight='balanced', median imputation inside pipeline
  - Note: imputation used for baseline only; not carried into final model or API

- **Final model (XGBoost, pruned):**
  - PR-AUC: 0.8691
  - Improvement over baseline: +0.4298
  - Parameters: n_estimators=10000, max_depth=6, learning_rate=0.05,
    scale_pos_weight=27.58, subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=200
  - Best iteration: 9987 / 10000
  - Convergence note: early stopping did not fire (hit n_estimators ceiling);
    gain in final 200 trees was 0.00016 PR-AUC — treated as converged.
  - Features: 422 (11 zero-importance features pruned before final training)
  - Zero-importance features dropped: V305, V107, V89, id_27, V88, V68, V65,
    V241, V41, V28, V14
  - Model size: 41.4 MB

- **Why XGBoost over LightGBM:** Not directly compared. XGBoost was chosen for
  its native NaN handling, mature Lambda deployment track record, and sufficient
  performance for this use case. LightGBM would be the first thing to try if
  training speed or model size became a constraint.

- **Why classical ML over deep learning:** Keeps the model artifact in the
  tens-of-MB range, avoids Lambda cold-start pain from loading large models,
  and requires explicit feature engineering decisions rather than delegating
  representation learning to a network.

- **Hyperparameter tuning:** Not run beyond the initial parameters. At 9987 trees
  to convergence, each RandomizedSearchCV trial would take 30–40 minutes.
  The marginal gain from tuning an already-converged model at PR-AUC 0.8691
  does not justify the compute time for a portfolio project.

- **Train/test split methodology:** Random stratified split (not time-based).
  The Kaggle competition used a time-based holdout, which is harder — PR-AUC on
  a true future holdout would likely be lower. Acceptable for a portfolio project;
  in production, time-based validation would be mandatory.

---

## What the Model Gets Wrong

- **23.8% of fraud is missed at threshold 0.5 (984 / 4,133 on test set)**
- **The model is not uncertain about these — it is confidently wrong.**
  Median predicted probability for missed fraud: 0.038. These are not borderline
  cases near the decision threshold. The model has assigned them high confidence
  of being legitimate. Threshold tuning cannot recover them.
- **Missed fraud skews higher-value:** median transaction amount for missed fraud
  is $97.00 vs $67.07 for caught fraud. The model disproportionately fails on
  larger transactions — the highest-cost cases in business terms.
- **Likely cause:** these transactions probably have unusual or sparse feature
  patterns not well-represented in training data — either novel fraud patterns,
  or fraud occurring in the product/identity segments with high missingness.
- **What I'd investigate next:** compare C-column and V-column distributions for
  the false negative cohort against true positives, and check whether they cluster
  in specific ProductCD values. Feature engineering targeting those segments would
  be the next lever.

---

## Threshold Decision

- **Operating threshold:** 0.0957
- **Recall target:** 85%
- **Actual recall:** 85.0%
- **Actual precision:** 67.0%
- **Missed fraud (FN):** 620 / 4,133 on test set
- **False alarms (FP):** 1,732 / 113,975 on test set

- **Business rationale:** A missed fraud costs the full transaction value plus
  chargeback fees (industry standard: $15–100 per disputed transaction depending
  on card network and merchant tier, per Visa/Mastercard operating regulations).
  On this dataset's median transaction of ~$75, a missed fraud costs $90–175
  minimum. A false positive costs one manual review action — roughly 10–15 minutes
  of analyst time, or approximately $5–15. That is a 10–15x cost asymmetry.
  Accepting 67% precision (1 in 3 flags is a false alarm) to achieve 85% recall
  is the correct tradeoff given those relative costs. Catching fraud is more
  important than review team efficiency.

- **Why not F1-optimal:** F1 maximisation treats false negatives and false
  positives as equal cost. They are not. F1 is the wrong objective function
  for this business problem.

- **Why not 80% recall:** Dropping from 85% to 80% recall would miss 207
  additional fraud cases to reduce false alarms from 1,732 to 660. That trades
  207 fraud losses for 1,072 fewer review actions — approximately 5 review
  actions saved per fraud case abandoned. Not worth it given the cost asymmetry.

- **Why not 90% recall:** At 90% recall, precision drops to 39.1% and false
  alarms rise to 5,795. The review queue becomes operationally unviable — the
  team would spend 60% of its time investigating legitimate transactions.

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
  calls/month) and run indefinitely at near-zero cost after credits expire.

- **Why not SageMaker:** Same credit-pool problem; SageMaker's own free-tier
  allowances were designed for the old 12-month free model and are unreliable
  on new accounts.

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