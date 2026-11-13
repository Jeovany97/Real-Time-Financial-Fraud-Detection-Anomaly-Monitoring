import numpy as np
import shap


def make_explainer(xgb_model):
    return shap.TreeExplainer(xgb_model)


def top_contributions(explainer, X_row, k=5):
    """Top-k SHAP contributions (log-odds) for a single-row DataFrame."""
    sv = np.asarray(explainer.shap_values(X_row))[0]
    idx = np.argsort(-np.abs(sv))[:k]
    cols = list(X_row.columns)
    return [{"feature": cols[i], "value": float(X_row.iloc[0, i]), "shap": float(sv[i])} for i in idx]


def global_summary_png(explainer, X_sample, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sv = explainer.shap_values(X_sample)
    shap.summary_plot(sv, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
