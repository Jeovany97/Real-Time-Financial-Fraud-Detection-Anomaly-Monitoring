"""uvicorn api.main:app --reload"""
import json, os, threading, time
from typing import Optional
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import create_model

from src.features import FEATURE_COLS, V_COLS
from src.explain import make_explainer, top_contributions

ART = os.getenv("ARTIFACT_DIR", "artifacts")
LOG = os.getenv("SCORED_LOG", f"{ART}/scored_log.jsonl")

app = FastAPI(title="Fraud Scoring API")
bundle = joblib.load(f"{ART}/xgb_bundle.joblib")
model, THRESH = bundle["model"], bundle["threshold"]
explainer = make_explainer(model)
featurizer = joblib.load(f"{ART}/featurizer_state.joblib")
lock = threading.Lock()  # featurizer state must be updated in arrival order

Transaction = create_model(
    "Transaction",
    transaction_id=(Optional[str], None),
    true_label=(Optional[int], None),  # only for simulation/monitoring
    Time=(float, ...), Amount=(float, ...),
    **{c: (float, ...) for c in V_COLS},
)


def risk_level(p):
    return "high" if p >= THRESH else "medium" if p >= THRESH * 0.5 else "low"


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESH}


@app.post("/score")
def score(tx: Transaction):
    t0 = time.perf_counter()
    d = tx.model_dump()
    with lock:
        beh = featurizer.transform_and_update(d["Time"], d["Amount"])
    X = pd.DataFrame([{**{c: d[c] for c in V_COLS}, **beh}])[FEATURE_COLS]
    p = float(model.predict_proba(X)[0, 1])
    alert = p >= THRESH
    explanation = top_contributions(explainer, X, k=5) if alert else []
    out = {
        "transaction_id": d["transaction_id"], "Time": d["Time"], "Amount": d["Amount"],
        "fraud_probability": p, "alert": alert, "risk_level": risk_level(p),
        "top_features": explanation, "true_label": d["true_label"],
        "latency_ms": (time.perf_counter() - t0) * 1000, "scored_at": time.time(),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


@app.post("/reset")
def reset():
    """Restore featurizer state to the val/test boundary and clear the log."""
    global featurizer
    with lock:
        featurizer = joblib.load(f"{ART}/featurizer_state.joblib")
        open(LOG, "w").close()
    return {"status": "reset"}
