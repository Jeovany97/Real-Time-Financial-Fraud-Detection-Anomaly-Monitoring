"""Train + compare 4 models on a strict chronological split.

python -m src.train --csv data/creditcard.csv --max-fpr 0.001
"""
import argparse, json, os
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import load, build_features, chronological_split, split_points
from .explain import make_explainer, global_summary_png

ART = "artifacts"


def pick_threshold(y, s, max_fpr):
    """On VALIDATION data: highest-recall threshold with FPR <= budget."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= max_fpr)[0]
    return float(thr[ok[np.argmax(tpr[ok])]])


def evaluate(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)),
        "alerts_per_10k": float(pred.mean() * 1e4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(thr),
    }


def main(csv, max_fpr):
    os.makedirs(ART, exist_ok=True)
    df = load(csv)
    a, b = split_points(len(df))
    X, snap = build_features(df, snapshot_at=b)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(df, X)
    print(f"train {len(Xtr)} ({ytr.mean():.4%} fraud) | val {len(Xva)} ({yva.mean():.4%}) | test {len(Xte)} ({yte.mean():.4%})")

    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="aucpr",
                                 early_stopping_rounds=30, n_jobs=-1, random_state=42),
        "isolation_forest": IsolationForest(n_estimators=300, contamination="auto", n_jobs=-1, random_state=42),
    }

    def score(name, m, X_):
        return -m.score_samples(X_) if name == "isolation_forest" else m.predict_proba(X_)[:, 1]

    metrics, test_scores, val_scores = {}, pd.DataFrame({"Time": df["Time"].values[b:], "Class": yte}), {}
    for name, m in models.items():
        print(f"fitting {name} ...")
        if name == "xgboost":
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        elif name == "isolation_forest":
            m.fit(Xtr)  # unsupervised: no labels used
        else:
            m.fit(Xtr, ytr)
        sva, ste = score(name, m, Xva), score(name, m, Xte)
        thr = pick_threshold(yva, sva, max_fpr)  # chosen on val, applied to test
        metrics[name] = evaluate(yte, ste, thr)
        test_scores[name] = ste
        val_scores[name] = sva
        print(f"  PR-AUC {metrics[name]['pr_auc']:.3f} | P {metrics[name]['precision']:.3f} | "
              f"R {metrics[name]['recall']:.3f} | FPR {metrics[name]['fpr']:.4%}")

    xgb = models["xgboost"]
    joblib.dump({"model": xgb, "threshold": metrics["xgboost"]["threshold"]}, f"{ART}/xgb_bundle.joblib")
    joblib.dump(snap, f"{ART}/featurizer_state.joblib")  # state at val/test boundary
    np.save(f"{ART}/val_scores.npy", val_scores["xgboost"])
    test_scores.to_csv(f"{ART}/test_scores.csv", index=False)
    json.dump({"max_fpr_budget": max_fpr, "models": metrics}, open(f"{ART}/metrics.json", "w"), indent=2)
    df.iloc[b:].to_csv(f"{ART}/test_stream.csv", index=False)  # raw rows for the simulator

    n = min(2000, len(Xte))
    global_summary_png(make_explainer(xgb), Xte.sample(n, random_state=0), f"{ART}/shap_summary.png")
    print("done -> artifacts/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/creditcard.csv")
    p.add_argument("--max-fpr", type=float, default=0.001, help="false-positive-rate budget")
    a = p.parse_args()
    main(a.csv, a.max_fpr)
