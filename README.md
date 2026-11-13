# Real-Time Financial Fraud Detection & Anomaly Monitoring

Python · scikit-learn · XGBoost · SHAP · FastAPI · Streamlit

Detects fraudulent card transactions on the [Kaggle ULB credit card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.17% fraud) and serves scores in a simulated real-time stream.

## Architecture
```
creditcard.csv ─► causal features ─► chronological split 60/20/20 ─► 4 models ─► metrics + artifacts
                                                                          │
simulator (replays FUTURE test period) ─► FastAPI /score ─► scored_log.jsonl ─► Streamlit dashboard
                                              │ XGBoost + SHAP explanation on every alert
```

## Run
```bash
pip install -r requirements.txt
# download creditcard.csv from Kaggle into data/
python -m src.train --csv data/creditcard.csv --max-fpr 0.001     # trains + evaluates 4 models
uvicorn api.main:app --port 8000                                  # terminal 2
python -m simulator.stream --rate 20 --reset                      # terminal 3
streamlit run dashboard/app.py                                    # terminal 4
```
Score a single transaction: `POST /score` with `Time`, `Amount`, `V1..V28` (optional `transaction_id`, `true_label`).

## Design decisions (good interview material)
- **Time-based split, no leakage.** Rows are sorted by `Time`; train = first 60%, validation = next 20%, test = final 20%.
  Behavioural features (txn velocity over 60s/600s, time since previous txn, amount z-score vs. *past* amounts) are
  computed by one streaming class that updates its state *after* emitting features. The same class powers training and
  the API, so there is no train/serve skew. The API starts from the featurizer state at the val/test boundary.
- **Imbalance.** `class_weight` (LR, RF), `scale_pos_weight` (XGBoost), unsupervised Isolation Forest as an anomaly baseline.
  Headline metric is PR-AUC, not accuracy/ROC-AUC.
- **Operating point as a business constraint.** Each model's threshold is chosen on *validation* to maximise recall under a
  false-positive-rate budget (default 0.1%), then applied untouched to the test set.
- **Explainability.** SHAP TreeExplainer: global summary plot plus top 5 per alert contributions returned by the API and
  shown in the dashboard's investigation tab.
- **Monitoring.** Alert rate, latency, running precision/recall (when labels arrive), and score drift via PSI vs. validation.

## Limitations (state these honestly)
- The dataset has no account/card ID, so behavioural features are global stream features, not per customer profiles.
- V1–V28 are PCA anonymised, so SHAP names components rather than human readable reasons.
- Only ~2 days of data and ~490 frauds total, so test set metrics are noisy (report counts alongside rates).
- Simulator is a replay, not true production traffic.

## Results (real data, chronological test set: 56,962 txns, 75 frauds)
Threshold per model picked on validation at FPR <= 0.1%.

| Model | PR-AUC | ROC-AUC | Precision | Recall | FPR |
|---|---|---|---|---|---|
| Logistic Regression | 0.745 | 0.982 | 0.750 | 0.720 | 0.032% |
| Random Forest | 0.815 | 0.972 | 0.694 | 0.787 | 0.046% |
| XGBoost | 0.759 | 0.983 | 0.683 | 0.747 | 0.046% |
| Isolation Forest (unsupervised) | 0.042 | 0.956 | 0.000 | 0.000 | 0.000% |

With only 75 test frauds, one fraud is ~1.3 points of recall, so differences between the top three models are within noise.
Isolation Forest ranks well (ROC-AUC 0.96) but its top-scored points are mostly legitimate outliers, so it catches nothing at the FPR budget.
