"""
detector.py — unsupervised anomaly detection over PON telemetry using
Isolation Forest.

Why Isolation Forest and why unsupervised:
Real ISPs don't have labeled "this was a tap" data — attacks are rare and
often go unnoticed for a long time (that's the whole problem this project
is about). So the model is trained ONLY on data that looks statistically
normal, and flags anything that isolates easily in the feature space —
same principle GPON-Guard v1's z-score rule used, but now multivariate and
able to pick up combinations of features a single z-score threshold misses
(e.g. "power dropped a little AND variance got unusually low" — the tap
signature — vs "power dropped a lot AND is fluctuating" — the bend
signature).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "power_dbm", "power_roll_mean", "power_roll_std",
    "power_delta_1", "power_delta_5", "power_dev_from_baseline",
    "temp_roll_std", "volt_roll_std",
]


def train(feat_df: pd.DataFrame, contamination: float = 0.05):
    """Train on rows the simulator marked as label==0 (normal) only —
    mirrors real deployment where you calibrate on a known-healthy baseline period."""
    normal = feat_df[feat_df["label"] == 0]
    X_train = normal[FEATURE_COLS].values

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train_scaled)
    return model, scaler


def score(feat_df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    X = feat_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    raw_score = model.decision_function(X_scaled)  # higher = more normal
    pred = model.predict(X_scaled)                  # -1 = anomaly, 1 = normal

    result = feat_df.copy()
    result["anomaly_score"] = -raw_score  # flip so higher = MORE anomalous (more intuitive)
    result["is_anomaly"] = (pred == -1).astype(int)
    return result


def evaluate(result: pd.DataFrame):
    """Compare flagged anomalies against the ground-truth event labels the
    simulator injected, per attack type."""
    print("\n--- Detection summary by true event type ---")
    for label_val, name in [(0, "normal"), (1, "macrobend (benign)"),
                             (2, "eavesdrop_tap"), (3, "brute_force")]:
        subset = result[result["label"] == label_val]
        if len(subset) == 0:
            continue
        flag_rate = subset["is_anomaly"].mean()
        print(f"  {name:24s}  n={len(subset):5d}   flagged={flag_rate:.1%}")


if __name__ == "__main__":
    feat = pd.read_csv("data/features.csv")
    model, scaler = train(feat)
    result = score(feat, model, scaler)
    result.to_csv("data/scored.csv", index=False)
    evaluate(result)

    joblib.dump(model, "models/isolation_forest.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("\nSaved model -> models/isolation_forest.joblib")
