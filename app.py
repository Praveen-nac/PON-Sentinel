"""
app.py — Flask API + live dashboard for PON-Sentinel.
Serves scored + explained anomaly results (precomputed via detector.py /
explain.py) so the dashboard loads instantly for a demo.
"""

import json
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template

from detector import score, FEATURE_COLS
from explain import explain_flagged

app = Flask(__name__)

MODEL = joblib.load("models/isolation_forest.joblib")
SCALER = joblib.load("models/scaler.joblib")
FEAT_DF = pd.read_csv("data/features.csv")
RESULT_DF = score(FEAT_DF, MODEL, SCALER)

EVENT_NAMES = {0: "normal", 1: "macrobend", 2: "eavesdrop_tap", 3: "brute_force"}


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    summary = []
    for label_val, name in EVENT_NAMES.items():
        subset = RESULT_DF[RESULT_DF["label"] == label_val]
        if len(subset) == 0:
            continue
        summary.append({
            "event_type": name,
            "count": int(len(subset)),
            "flagged_pct": round(float(subset["is_anomaly"].mean()) * 100, 1),
        })
    return jsonify(summary)


@app.route("/api/onus")
def api_onus():
    onu_summary = []
    for onu_id, g in RESULT_DF.groupby("onu_id"):
        anomaly_pts = int(g["is_anomaly"].sum())
        true_event = g[g["label"] != 0]["label"].mode()
        true_event_name = EVENT_NAMES[int(true_event.iloc[0])] if len(true_event) else "normal"
        onu_summary.append({
            "onu_id": int(onu_id),
            "anomaly_points": anomaly_pts,
            "max_anomaly_score": round(float(g["anomaly_score"].max()), 4),
            "true_event": true_event_name,
        })
    onu_summary.sort(key=lambda x: -x["max_anomaly_score"])
    return jsonify(onu_summary)


@app.route("/api/onu/<int:onu_id>")
def api_onu_detail(onu_id):
    g = RESULT_DF[RESULT_DF["onu_id"] == onu_id].sort_values("t")
    return jsonify({
        "onu_id": onu_id,
        "t": g["t"].tolist(),
        "power_dbm": g["power_dbm"].tolist(),
        "anomaly_score": g["anomaly_score"].round(4).tolist(),
        "is_anomaly": g["is_anomaly"].tolist(),
        "label": g["label"].tolist(),
    })


@app.route("/api/explain/<int:onu_id>")
def api_explain(onu_id):
    g = RESULT_DF[(RESULT_DF["onu_id"] == onu_id) & (RESULT_DF["is_anomaly"] == 1)].copy()
    if g.empty:
        return jsonify({"onu_id": onu_id, "reasons": []})
    # cap for demo speed — explain the top few most anomalous points for this ONU
    g = g.sort_values("anomaly_score", ascending=False).head(5)
    explained = explain_flagged(g, MODEL, SCALER)
    reasons = [
        {"t": int(r.t), "anomaly_score": round(float(r.anomaly_score), 4),
         "top_reasons": r.top_reasons}
        for r in explained.itertuples()
    ]
    return jsonify({"onu_id": onu_id, "reasons": reasons})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
