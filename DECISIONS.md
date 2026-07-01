# Design Decisions Log

---

## Problem & Dataset

**Problem:** Predict whether a card-not-present e-commerce transaction is fraudulent, framed as flagging high-risk transactions for manual review rather than emitting a raw probability score.

**Dataset:** IEEE-CIS Fraud Detection (Kaggle, 2019), provided by Vesta Corporation.
Source: kaggle.com/c/ieee-fraud-detection
~590,540 labeled transactions, 431 raw features (400 numerical, 31 categorical), ~3.5% fraud rate. Two tables requiring a join — train_transaction and train_identity. Identity coverage is partial; not every transaction has a matching identity record.

**Why this dataset over ULB or PaySim:**
ULB's features are pre-PCA-transformed. Inputs are anonymous components, not real fields, which removes any meaningful feature engineering and makes a business-decision-style demo impossible. PaySim is synthetic — simulator output, not actual transactions. IEEE-CIS is real, messy (hundreds of sparse columns, a two-table join where 75.6% of rows have no identity match), and has interpretable fields (amount, product code, card network, device type, email domain). That's what makes a real demo possible.

**Why real-time, not batch:** A fraud/no-fraud decision is needed at transaction authorisation. The model is small enough to serve synchronously with acceptable latency.

---

## Evaluation Metric
*Logged before model training.*

**Primary:** PR-AUC (Precision-Recall Area Under Curve)
**Secondary:** F1 on the fraud class at the chosen operating threshold

**Why not accuracy:** At 3.5% fraud rate, predicting "not fraud" for every transaction scores 96.5% accuracy. The metric rewards ignoring the task entirely.

**Why PR-AUC over ROC-AUC:** ROC-AUC is optimistic under class imbalance — it counts true negatives, which are easy to get right when they're 96.5% of the data. PR-AUC focuses on the minority class and is more honest about what the model is actually doing.

**Why PR-AUC over a single F1:** F1 at a fixed threshold depends heavily on which threshold you pick. PR-AUC evaluates across all thresholds and gives a complete picture before committing to an operating point.

---

## EDA Findings

- Confirmed fraud rate: 3.499% (20,663 fraud / 569,877 legitimate out of 590,540)
- Identity coverage: 24.4% of transactions have identity records. 75.6% don't. This is the dominant inference-time case, not an edge case. The API treats missing identity fields as NaN.
- Total columns after join: 434
- Columns with any missing: 414
- Columns >50% missing: 214
- Columns >80% missing: 74

**D-column missingness as fraud signal:**
D6, D7, D12, D13, D14 are missing 25–27 percentage points less often in fraud than in legitimate transactions (D12: 63% missing for fraud vs 90% for legitimate). These are time-delta features — time since previous transaction or event. Fraudsters generate sequences; legitimate one-off buyers often have no prior history. Missingness here is signal, not noise. Dropping these columns would be an error.

**Time-of-day fraud pattern:**
Fraud rate peaks at 7am (10.6%) and drops to 2.3% at 1pm — roughly a 4.6x swing. This matches what you'd expect from card-not-present fraud: attacks cluster when cardholders aren't watching their accounts. hour_of_day and day_of_week_proxy were engineered from TransactionDT and kept as features.

**TransactionAmt:** Heavily right-skewed (mean $135, median $69, max $31,937). Fraud transactions have a higher median ($75) and mean ($149 vs $135). log1p transform applied.

**Feature importance — final model:**
V258 (24.7%), V201 (8.6%), V70 (3.7%), V122 (2.3%), V294 (2.1%). TransactionID and TransactionDT did not appear in the leakage audit or final importances. Leakage audit passed.

---

## Join Strategy

Left join on TransactionID — every transaction keeps its row; identity fields are NaN where no match exists.

Inner join was rejected because dropping identity-absent rows would bias the training set. Whether those transactions have a systematically different fraud rate is unknown — removing them would mean assuming the answer.

Inference-time implication: the API accepts requests where identity fields are absent and treats them as NaN. Missing identity features are not errors.

---

## Missing Value Strategy

No imputation. No sentinel filling (-999). No column dropping based on missingness rate. NaN passes directly to XGBoost.

XGBoost learns the optimal direction to route missing values at each split during training. Filling -999 treats missingness as a real numeric value and corrupts that. The D-column missingness differences (25+ percentage points between fraud and legitimate) confirm empirically that dropping high-missingness columns throws away signal.

Exception: the logistic regression baseline can't handle NaN. Median imputation was applied inside a pipeline used only for the baseline — it is not reused at inference time.

---

## Class Imbalance Strategy

- Class ratio: 569,877 legitimate / 20,663 fraud = 27.58:1
- Approach: `scale_pos_weight=27.58` in XGBoost, computed from y_train only (not the full dataset — computing from the full dataset would be leakage)
- SMOTE was rejected: it adds complexity and can distort high-dimensional feature spaces. Class weighting achieves the same loss-function correction without synthetic data generation.
- Threshold tuning applied post-training based on business cost asymmetry — the operating threshold is not 0.5.

---

## TransactionAmt Transformation

log1p applied at both training time and inference time.

Raw distribution is heavily right-skewed (max $31,937). log1p handles zero-value transactions cleanly, unlike log. The transform is applied in scripts/preprocess.py and called identically at inference time. A transform applied only at training time is silent leakage in reverse — the model would run inference on a different scale than it was trained on.

---

## Categorical Encoding

pandas category codes (.astype('category').cat.codes), with -1 (the NaN code) replaced by NaN so XGBoost handles missing categoricals natively.

Columns encoded: ProductCD, card4, card6, P_emaildomain, R_emaildomain, and id_ string columns.

One-hot encoding was rejected: email domains alone have hundreds of unique values. One-hot would explode dimensionality. XGBoost handles integer-encoded categoricals natively.

The integer-to-string mappings are saved as model/category_maps.json. Without this file, preprocess.py cannot replicate the same encoding on a single transaction at inference time — .cat.codes assigns integers based on the sorted unique values in the training data at that moment, and that ordering only exists in memory during the notebook run.

---

## TransactionDT — Time Feature Engineering

TransactionDT is a seconds offset from an undisclosed Vesta reference point. Raw, it's a monotonically increasing index — not a useful feature. It's dropped from the feature set.

Engineered features retained:
- hour_of_day: (TransactionDT % 86400) / 3600
- day_of_week_proxy: (TransactionDT // 86400) % 7

Fraud rate at 7am is 10.6% vs 2.3% at 1pm. Dropping TransactionDT without analysis was an error in the first notebook version. It left 4.6x of signal on the table.

---

## Model Selection

**Baseline (Logistic Regression):**
- PR-AUC: 0.4393
- Median imputation inside a sklearn Pipeline, used for the baseline only — not carried into the final model or API

**Final model (XGBoost):**
- PR-AUC: 0.8691, improvement of +0.4298 over baseline
- Parameters: n_estimators=10000, max_depth=6, learning_rate=0.05, scale_pos_weight=27.58, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=200
- Best iteration: 9987/10000. Early stopping did not fire — the model hit the n_estimators ceiling. Gain in the final 200 trees was 0.00016 PR-AUC. Treated as converged.
- 11 zero-importance features pruned before final training: V305, V107, V89, id_27, V88, V68, V65, V241, V41, V28, V14
- Final feature count: 422
- Model size: 41.4 MB (saved as model.ubj)

**Why XGBoost over LightGBM:** Not directly compared. XGBoost was chosen for its native NaN handling and mature Lambda deployment history. LightGBM would be the first thing to benchmark if training speed or model size became a problem.

**Why classical ML over deep learning:** Keeps the artifact small, avoids Lambda cold-start pain, and requires real feature engineering decisions rather than handing the problem to a network.

**Hyperparameter tuning:** Not run beyond the initial fixed parameters. At 9987 trees to convergence, each RandomizedSearchCV trial would take 30–40 minutes. The marginal gain from tuning an already-converged model at PR-AUC 0.8691 doesn't justify that for a portfolio project.

**Train/test split:** Random stratified split, not time-based. The Kaggle competition used a time-based holdout, which is harder — PR-AUC on a true future holdout would likely be lower. Acceptable here; in production, time-based validation would be mandatory.

---

## What the Model Gets Wrong

At threshold 0.5, 984 fraud cases are missed out of 4,133 on the test set (23.8%).

These aren't uncertain predictions near the boundary. Median predicted probability for missed fraud is 0.038 — the model is confidently calling them legitimate. Threshold tuning can't recover them.

Missed fraud skews higher-value: median transaction amount for misses is $97 vs $67 for correctly caught fraud. The model disproportionately fails on larger transactions — the highest-cost failures in business terms.

Most likely cause: these transactions have unusual or sparse V-feature patterns not well-represented in training — either novel fraud patterns, or cases concentrated in product/identity segments with high missingness.

What I'd check next: compare C-column and V-column distributions for the false negative cohort against true positives, and look at whether they cluster in specific ProductCD values.

---

## Threshold Decision

- Operating threshold: 0.0957
- Target recall: 85%
- Actual recall: 85.0%, actual precision: 67.0%
- False negatives: 620 / 4,133 on test set
- False positives: 1,732 / 113,975 on test set

A missed fraud costs the transaction value plus chargeback fees ($15–100 per disputed transaction, per Visa/Mastercard operating regulations). On this dataset's median transaction of ~$75, that's $90–175 minimum per miss. A false positive costs one manual review — roughly 10–15 minutes of analyst time, or $5–15. That's a 10–15x cost asymmetry.

F1-optimal was rejected: F1 maximisation treats false negatives and false positives as equal cost. They're not.

80% recall rejected: dropping to 80% would miss 207 more fraud cases to reduce false alarms from 1,732 to 660. That's roughly 5 review actions saved per fraud case abandoned. Not worth it.

90% recall rejected: precision drops to 39.1% and false alarms hit 5,795. The review queue becomes unworkable — the team spends the majority of its time on legitimate transactions.

---

## Architecture

- Inference: AWS Lambda (container image via ECR) + API Gateway HTTP API
- Model storage: S3 (artifact versioning, separate from the container)
- Logging: CloudWatch (Lambda logs) + DynamoDB (prediction log, Phase 7 — not yet built)
- Frontend: Streamlit on Streamlit Community Cloud (not AWS)

Lambda and API Gateway sit on AWS's permanent Always Free allowances (1M Lambda requests/month, 1M API Gateway calls/month). EC2 and SageMaker draw down the credit balance on new AWS accounts (post-mid-2025), which expires after six months. This architecture stays live after the credits run out.

SageMaker was also rejected: same credit-pool problem, and its free-tier allowances were designed for the old 12-month free model.

Cold starts are a known limitation — Lambda adds latency on the first request after an idle period. Acceptable for a portfolio demo. Provisioned concurrency would fix it in production.

The Streamlit frontend is hosted separately to remove all billing risk from the UI layer. It's a deliberate infrastructure decision.

---

## IAM & AWS Setup

- IAM user: abhinavtadi-dev
- Policies: AmazonEC2ContainerRegistryFullAccess, AWSLambda_FullAccess, AmazonAPIGatewayAdministrator, AmazonS3FullAccess, CloudWatchLogsFullAccess
- Broader than least-privilege. In production, permissions would be scoped to exactly what each service needs.
- Root account: MFA enabled, no access keys generated, not used for any resource creation
- Region: us-east-1. Originally set to ap-south-2 (Hyderabad), which is an opt-in region — STS calls returned InvalidClientTokenId even with valid credentials. Switched to us-east-1.
- Billing alert: $1 monthly cost budget, 100% actual-spend alert, email configured before any resource was created

---

## Container Architecture & Base Image

Building for linux/arm64 — Docker Desktop on Apple Silicon defaults to the host architecture. This has one hard consequence at deployment: the Lambda function must be created with `--architectures arm64`. Lambda's default is x86_64. An arm64 image pushed to an x86_64 function deploys without error and fails at invoke time — not at deployment, at invocation. Easy to miss if you don't know to look for it.

Base image: switched from `public.ecr.aws/lambda/python:3.11` (Amazon Linux 2, glibc 2.26) to `public.ecr.aws/lambda/python:3.12` (Amazon Linux 2023, glibc 2.34). The 3.11 image produced an XGBoost glibc warning at every cold start — XGBoost dropped support for glibc < 2.28 after May 2025. The 3.12 image eliminates the warning. All pinned packages resolved as prebuilt aarch64 wheels on 3.12 with no source compilation needed.

---

## Model Serialization Format

Changed from joblib (pickle-based) to XGBoost's native `save_model()` / `load_model()`, saving as model.ubj (Universal Binary JSON).

joblib pickles the entire sklearn wrapper. XGBoost's own documentation warns against using pickle across versions — the artifact is tied to the exact XGBoost and sklearn versions at training time, and loading under different versions either produces a warning or, in worse cases, silently wrong behaviour. `.ubj` is XGBoost's recommended durable format and is also smaller than the pickle.

The explicit `iteration_range=(0, _BEST_ITERATION + 1)` in inference.py is unaffected by the format change — that guard lives in code.

---

## Mangum Local Test Event Format

Mangum 0.17.0 inspects the event structure to determine which handler to use. The v1 test payload format — `httpMethod` at the top level, no `requestContext` — doesn't match any recognized pattern and throws `RuntimeError: unable to infer a handler`.

The correct format for local container testing is HTTP API v2: `version: "2.0"` and `routeKey` at the top level, with the HTTP method nested inside `requestContext.http.method`. This matches the actual event format that API Gateway HTTP API sends in production — not a workaround, just the right format.

---

## Monitoring

DynamoDB prediction logging is planned for Phase 7 but not yet built. Each prediction will log: request UUID, input features, prediction, probability score, timestamp.

Automated drift detection is not built.

What I'd build next: a scheduled Lambda (EventBridge, weekly) that queries the last 7 days of DynamoDB logs, computes the distribution of key inputs (TransactionAmt, card type, device type), and compares against the training distribution using a KS test or Population Stability Index (PSI > 0.2 threshold). Alerts via SNS on significant shift or if the predicted fraud rate moves more than 2 standard deviations from the training baseline.

The reason it's not in v1 is time, not difficulty.