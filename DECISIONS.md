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
*Logged before model training.* (27-06-2026)

- **Primary metric:** PR-AUC (Precision-Recall Area Under Curve)
- **Secondary metric:** F1 score on the fraud class (minority class) at the chosen
  operating threshold

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

## Join Strategy (train_transaction + train_identity)

- **Approach:** Left join on TransactionID — every transaction gets a row; identity
  fields are NaN where no matching identity record exists (~60% of transactions).

- **Why not an inner join:** Dropping transactions without identity records would bias
  the training set. We do not know whether identity-absent transactions have a
  systematically different fraud rate — discarding them would be assuming the answer.

- **Inference-time implication:** The API must accept requests where identity fields
  are absent and treat them as NaN, exactly as they appear in training data. The
  preprocessing function must handle this explicitly — missing identity features are
  not errors, they are a known input state. This must be validated during Phase 2
  local testing before deployment.

---

## Class Imbalance Strategy
*To be finalised during Phase 1 after EDA.*

- **Candidates:** `scale_pos_weight` in XGBoost (ratio of negative to positive class),
  SMOTE oversampling, threshold tuning post-training, or a combination.

- **Working hypothesis:** `scale_pos_weight` combined with threshold tuning is
  sufficient. SMOTE adds complexity and can distort the feature space on high-
  dimensional tabular data without a clear accuracy benefit at this scale.

- **Will update:** chosen approach, the specific parameter value used, and the
  reasoning for preferring it over alternatives — after EDA confirms the actual
  class distribution in the training split.

---

## Feature Engineering
*To be documented during Phase 1 EDA. Template for each entry:*
`[Feature or group]: [transformation applied] — because [reason observed in EDA].`

*Entries will be added here as decisions are made. Do not reconstruct these from
memory after training — log them at the time.*

---

## Model Selection

- **Baseline:** Logistic Regression (scikit-learn, `class_weight='balanced'`).
  Required comparison point — the final model's improvement needs to be explained
  relative to something, not just stated as a number.

- **Target model:** XGBoost classifier. Hypothesis to be confirmed after baseline
  comparison.

- **Why classical ML over deep learning:** Keeps the model artifact in the tens-of-MB
  range, avoids Lambda cold-start latency from loading large model files, and requires
  explicit feature engineering decisions rather than delegating representation learning
  to a network — which produces more defensible interview answers.

- **Why XGBoost over LightGBM:** To be filled after direct comparison during Phase 1,
  if a comparison is run. If only one is trained, document why the other was not
  evaluated.

- **Hyperparameter tuning:** `RandomizedSearchCV`. Not full `GridSearchCV` — this is
  not a Kaggle leaderboard optimisation. A bounded random search is sufficient to
  demonstrate the practice and find reasonable parameters without overfitting to the
  test set through exhaustive search.

- **Model size constraint:** Target < 50 MB for `model.joblib`. If the trained model
  exceeds this, reduce `n_estimators` and `max_depth` before treating training as
  complete. A 200 MB model is a deployment problem, not a better model.

- **What the model gets wrong:** To be documented after evaluation on the held-out
  test set. Specifically: what types of transactions get misclassified, and in which
  direction (false positive vs false negative). This section must be filled before
  the project is called done — it is a required interview answer.

---

## Architecture

- **Inference:** AWS Lambda (container image via ECR) + API Gateway HTTP API
- **Model storage:** S3 (artifact versioning, separate from the container)
- **Logging:** DynamoDB (prediction log) + CloudWatch (Lambda logs)
- **Frontend:** Streamlit, hosted on Streamlit Community Cloud (not AWS)

- **Why Lambda + API Gateway over EC2:**
  AWS accounts created post-mid-2025 use a credit-based free tier (~$100–200,
  expires after six months or when exhausted). EC2 draws down that credit balance
  and generates surprise charges when left running. Lambda and API Gateway sit on
  AWS's permanent Always Free allowances (1M Lambda requests/month, 1M API Gateway
  calls/month) and run indefinitely at near-zero cost after credits expire. This
  architecture keeps the project live for the portfolio long-term without
  babysitting a bill.

- **Why not SageMaker:** Same credit-pool problem; SageMaker's own free-tier
  allowances were designed for the old 12-month free model and are unreliable on
  new accounts.

- **Known simplification — cold starts:** Lambda adds latency on the first request
  after an idle period (typically 1–3 seconds for a container-image function).
  Acceptable for a portfolio demo; provisioned concurrency would eliminate this in
  a production setting.

- **Why the Streamlit frontend is hosted separately (Streamlit Community Cloud):**
  Hosting the UI layer on AWS would add cost risk with no architectural benefit
  for a demo. Streamlit Community Cloud is free, requires no AWS configuration,
  and keeps the AWS cost surface limited to inference only. This is a deliberate
  infrastructure decision, not a shortcut.

---

## IAM & AWS Setup

- **IAM user:** abhinavtadi-dev
- **Policies attached:** `AmazonEC2ContainerRegistryFullAccess`,
  `AWSLambda_FullAccess`, `AmazonAPIGatewayAdministrator`, `AmazonS3FullAccess`,
  `CloudWatchLogsFullAccess`
- **Known simplification:** These are broader than least-privilege. In a production
  setting, each service would be scoped to only the permissions it requires.
  Acknowledged and accepted for a first-project context.
- **Root account:** MFA enabled, no access keys generated, not used for any
  resource creation.
- **AWS Region:** `us-east-1`. Originally configured as `ap-south-2` (Hyderabad),
  which is an opt-in region requiring explicit account activation — STS calls
  returned `InvalidClientTokenId` against it even with valid credentials. Switched
  to `us-east-1` for full service coverage and alignment with guide defaults.
- **Billing alert:** $1.00 monthly cost budget, 100% actual-spend alert, email
  notification configured.

---

## Monitoring

- **Implemented:** DynamoDB prediction logging (Phase 7 — planned, not yet built).
  Each prediction will log: TransactionID or request UUID, input features,
  prediction output, probability score, timestamp.

- **Not implemented:** Automated drift detection.

- **What I would build next:** A scheduled Lambda (EventBridge trigger, weekly) that
  queries the last 7 days of DynamoDB prediction logs, computes the distribution of
  key input features (TransactionAmt, card type, device type), and compares against
  the training distribution using a KS test or Population Stability Index (PSI).
  Alerts via SNS if PSI > 0.2 on any key feature or if the predicted fraud rate
  shifts more than 2 standard deviations from the training baseline.

- **Why not built for v1:** Outside the scope of the current project timeline.
  Manual monitoring via CloudWatch inspection and periodic DynamoDB queries is
  sufficient to demonstrate the concept and answer interview questions honestly.
  The design above is what I would implement given another two to three weeks.