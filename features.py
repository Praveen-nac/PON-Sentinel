"""
features.py — turns raw per-timestep telemetry into feature vectors the
model can reason over. Rolling windows are what let the model see a SUDDEN
DROP vs a SLOW DRIFT vs a LOW-VARIANCE PLATEAU — the actual signatures that
separate a tap from a bend from normal noise.
"""

import numpy as np
import pandas as pd

WINDOW = 10  # rolling window size (timesteps)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for onu_id, g in df.groupby("onu_id"):
        g = g.sort_values("t").reset_index(drop=True)

        power = g["power_dbm"]
        temp = g["temp_c"]
        volt = g["volt"]

        feat = pd.DataFrame({
            "onu_id": g["onu_id"],
            "t": g["t"],
            "label": g["label"],

            # absolute levels
            "power_dbm": power,
            "temp_c": temp,
            "volt": volt,

            # rolling mean/std — captures plateau + stability signature
            "power_roll_mean": power.rolling(WINDOW, min_periods=1).mean(),
            "power_roll_std": power.rolling(WINDOW, min_periods=1).std().fillna(0),

            # short-horizon delta — captures suddenness of a change
            "power_delta_1": power.diff().fillna(0),
            "power_delta_5": power.diff(5).fillna(0),

            # deviation from each ONU's own long-run baseline (first 30 samples = "known good")
            "power_dev_from_baseline": power - power.iloc[:30].mean(),

            "temp_roll_std": temp.rolling(WINDOW, min_periods=1).std().fillna(0),
            "volt_roll_std": volt.rolling(WINDOW, min_periods=1).std().fillna(0),
        })
        out.append(feat)

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    raw = pd.read_csv("data/telemetry.csv")
    feat = build_features(raw)
    feat.to_csv("data/features.csv", index=False)
    print(f"Built {len(feat)} feature rows, {feat.shape[1]} columns")
    print(feat.columns.tolist())
