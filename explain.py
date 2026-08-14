"""
explain.py — explainability layer on top of the Isolation Forest detector.

Why this matters (this is the actual novelty over GPON-Guard v1):
A network operator won't trust a black-box "ANOMALY" flag. When the system
flags an ONU, it needs to say WHICH signal drove the decision — e.g.
"flagged mainly because of power_dev_from_baseline and low power_roll_std"
(the tap signature) vs "flagged mainly because of power_roll_std spiking"
(the bend signature). SHAP values give a per-feature contribution score
for each individual prediction, not just global feature importance.
"""

import joblib
import numpy as np
import pandas as pd
import shap

from detector import FEATURE_COLS


def build_explainer(model, background_X: np.ndarray):
    """KernelExplainer works for any model with a decision_function, at the
    cost of speed — fine here since we only explain FLAGGED rows, not all of them."""
    background = shap.sample(background_X, 50, random_state=42)
    explainer = shap.Explainer(model.decision_function, background)
    return explainer


def explain_flagged(result: pd.DataFrame, model, scaler, top_n_reasons: int = 3):
    flagged = result[result["is_anomaly"] == 1].copy()
    if flagged.empty:
        print("No anomalies flagged — nothing to explain.")
        return flagged

    X_all_scaled = scaler.transform(result[FEATURE_COLS].values)
    X_flag_scaled = scaler.transform(flagged[FEATURE_COLS].values)

    explainer = build_explainer(model, X_all_scaled)
    shap_values = explainer(X_flag_scaled)

    reasons = []
    for i in range(len(flagged)):
        row_shap = shap_values.values[i]
        # decision_function: LOWER = more anomalous, so the most negative
        # SHAP contributions are what PUSHED this row toward "anomaly"
        order = np.argsort(row_shap)[:top_n_reasons]
        top_features = [(FEATURE_COLS[j], round(float(row_shap[j]), 4)) for j in order]
        reasons.append(top_features)

    flagged["top_reasons"] = reasons
    return flagged


if __name__ == "__main__":
    result = pd.read_csv("data/scored.csv")
    model = joblib.load("models/isolation_forest.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # Explain a manageable sample so this runs fast for a demo (KernelExplainer is slow)
    sample = result[result["is_anomaly"] == 1].sample(
        min(30, (result["is_anomaly"] == 1).sum()), random_state=1
    )
    explained = explain_flagged(sample, model, scaler)

    pd.set_option("display.max_colwidth", None)
    print("\n--- Sample explained detections ---")
    print(explained[["onu_id", "t", "label", "anomaly_score", "top_reasons"]].to_string(index=False))

    explained.to_csv("data/explained_sample.csv", index=False)
    print("\nSaved -> data/explained_sample.csv")
