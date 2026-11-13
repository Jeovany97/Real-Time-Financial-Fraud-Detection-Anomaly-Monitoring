"""streamlit run dashboard/app.py"""
import json, time
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

ART = "artifacts"
st.set_page_config(page_title="Fraud Monitor", layout="wide")
st.title("Real-Time Fraud Detection & Anomaly Monitoring")
auto = st.sidebar.toggle("Auto-refresh (2s)", True)
tab_live, tab_models, tab_inv = st.tabs(["Live monitor", "Model comparison", "Alert investigation"])


def load_log():
    try:
        return pd.read_json(f"{ART}/scored_log.jsonl", lines=True)
    except Exception:
        return pd.DataFrame()


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


log = load_log()

with tab_live:
    if log.empty:
        st.info("No scored transactions yet. Start the API, then run `python -m simulator.stream --reset`.")
    else:
        alerts = log[log.alert]
        c = st.columns(5)
        c[0].metric("Transactions scored", f"{len(log):,}")
        c[1].metric("Alerts", f"{len(alerts):,}")
        c[2].metric("Alert rate", f"{log.alert.mean():.3%}")
        c[3].metric("p95 latency", f"{log.latency_ms.quantile(.95):.1f} ms")
        lab = log.dropna(subset=["true_label"])
        if len(lab) and lab.true_label.sum() > 0:
            tp = ((lab.alert) & (lab.true_label == 1)).sum()
            c[4].metric("Running precision / recall",
                        f"{tp / max(lab.alert.sum(), 1):.2f} / {tp / lab.true_label.sum():.2f}")

        l, r = st.columns(2)
        with l:
            st.subheader("Score distribution")
            st.plotly_chart(px.histogram(log, x="fraud_probability", color="alert", nbins=60, log_y=True),
                            use_container_width=True)
        with r:
            st.subheader("Rolling alert rate (last 200 tx)")
            roll = log.alert.astype(float).rolling(200, min_periods=20).mean()
            st.plotly_chart(px.line(roll, labels={"value": "alert rate", "index": "tx #"}),
                            use_container_width=True)

        try:
            ref = np.load(f"{ART}/val_scores.npy")
            if len(log) >= 500:
                st.metric("Score drift (PSI vs validation)",
                          f"{psi(ref, log.fraud_probability.tail(500).values):.3f}",
                          help="<0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift")
        except FileNotFoundError:
            pass

        st.subheader("Latest alerts")
        cols = ["transaction_id", "Time", "Amount", "fraud_probability", "risk_level", "true_label"]
        st.dataframe(alerts[cols].tail(25).iloc[::-1], use_container_width=True)

with tab_models:
    try:
        m = json.load(open(f"{ART}/metrics.json"))
        st.caption(f"Chronological 60/20/20 split. Threshold per model chosen on validation at "
                   f"FPR <= {m['max_fpr_budget']:.2%}; numbers below are on the held-out FUTURE test set.")
        tbl = pd.DataFrame(m["models"]).T[["pr_auc", "roc_auc", "precision", "recall", "fpr", "alerts_per_10k"]]
        st.dataframe(tbl.style.format("{:.4f}"), use_container_width=True)
        ts = pd.read_csv(f"{ART}/test_scores.csv")
        fig = go.Figure()
        for name in m["models"]:
            p, rc, _ = precision_recall_curve(ts.Class, ts[name])
            fig.add_scatter(x=rc, y=p, name=name, mode="lines")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision-Recall curves (test)")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Global SHAP importance (XGBoost)")
        st.image(f"{ART}/shap_summary.png")
    except FileNotFoundError:
        st.warning("Run `python -m src.train` first.")

with tab_inv:
    if log.empty or not log.alert.any():
        st.info("No alerts to investigate yet.")
    else:
        al = log[log.alert].tail(100).iloc[::-1].reset_index(drop=True)
        i = st.selectbox("Alert", range(len(al)),
                         format_func=lambda k: f"{al.transaction_id[k]}  (p={al.fraud_probability[k]:.3f})")
        row = al.iloc[i]
        contrib = pd.DataFrame(row.top_features).sort_values("shap")
        contrib["label"] = contrib.feature + " = " + contrib.value.round(3).astype(str)
        fig = px.bar(contrib, x="shap", y="label", orientation="h",
                     color=contrib.shap > 0, color_discrete_map={True: "#d62728", False: "#1f77b4"})
        fig.update_layout(showlegend=False, xaxis_title="SHAP value (log-odds toward fraud)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Amount **${row.Amount:,.2f}** | probability **{row.fraud_probability:.3f}** | "
                 f"true label: **{row.true_label}**")

if auto and not log.empty:
    time.sleep(2)
    st.rerun()
