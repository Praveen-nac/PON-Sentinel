"""
simulate.py — Synthetic GPON/EPON telemetry generator for PON-Sentinel

Generates realistic per-ONU optical power (dBm), temperature (C), and voltage (V)
time-series, with injected "normal degradation" events (macrobend, aging) AND
"eavesdropping / fiber-tap" events, so the two can be told apart by the model.

Why this distinction matters (this is the actual research problem):
A macrobend or connector dust causes a power drop that fluctuates and often
partially recovers. A fiber tap (someone splicing in an optical coupler to
siphon signal) causes a SMALL, SUDDEN, and PERSISTENT power drop with
unusually LOW variance afterward (because a fraction of light is being
siphoned off cleanly, not scattered randomly). That signature difference is
what the model is trained to pick up on — a plain threshold-on-power-drop
rule (what GPON-Guard v1 does) cannot reliably tell these apart.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ONUS = 32
N_TIMESTEPS = 500          # simulated telemetry samples per ONU (e.g. one per few minutes)
BASE_POWER_DBM = -18.0     # typical healthy GPON downstream received power
BASE_TEMP_C = 45.0
BASE_VOLT = 3.3


def _normal_series(length, base, noise_std, drift_std):
    """Slow random-walk drift + gaussian noise — how a healthy ONU behaves."""
    drift = np.cumsum(RNG.normal(0, drift_std, length))
    noise = RNG.normal(0, noise_std, length)
    return base + drift + noise


def _inject_macrobend(power, start, duration):
    """Benign fault: partial, fluctuating recovery (cable stress, temperature-induced bend)."""
    end = min(start + duration, len(power))
    dip = RNG.uniform(2.0, 5.0)
    recovery_noise = RNG.normal(0, 0.6, end - start)
    power[start:end] -= dip
    power[start:end] += np.linspace(0, dip * 0.4, end - start)  # partial recovery
    power[start:end] += recovery_noise
    return power


def _inject_eavesdrop_tap(power, start, duration):
    """
    Malicious fiber tap: small, sudden, persistent drop with LOW variance
    (a clean optical coupler siphons a fixed fraction of light — it doesn't
    introduce the random scattering a bend or dirty connector does).
    """
    end = min(start + duration, len(power))
    tap_loss = RNG.uniform(0.5, 1.8)   # subtle — designed to evade naive thresholds
    power[start:end] -= tap_loss
    power[start:end] += RNG.normal(0, 0.08, end - start)  # unusually stable/low-noise
    return power


def _inject_brute_force(temp, volt, start, duration):
    """OLT console brute-force: repeated auth attempts correlate with brief voltage/temp jitter
    on the management interface (proxy signal for control-plane load)."""
    end = min(start + duration, len(temp))
    temp[start:end] += RNG.normal(0.8, 0.3, end - start)
    volt[start:end] += RNG.normal(0, 0.05, end - start) * RNG.choice([1, -1], end - start)
    return temp, volt


def generate_dataset():
    rows = []
    event_log = []

    for onu_id in range(N_ONUS):
        power = _normal_series(N_TIMESTEPS, BASE_POWER_DBM, noise_std=0.35, drift_std=0.01)
        temp = _normal_series(N_TIMESTEPS, BASE_TEMP_C, noise_std=0.5, drift_std=0.01)
        volt = _normal_series(N_TIMESTEPS, BASE_VOLT, noise_std=0.02, drift_std=0.001)
        label = np.zeros(N_TIMESTEPS, dtype=int)  # 0 = normal, 1 = benign fault, 2 = eavesdrop, 3 = brute-force

        # ~35% of ONUs get an injected event somewhere in their timeline
        roll = RNG.random()
        if roll < 0.12:
            start = int(RNG.integers(50, N_TIMESTEPS - 80))
            dur = int(RNG.integers(20, 60))
            power = _inject_macrobend(power, start, dur)
            label[start:start + dur] = 1
            event_log.append((onu_id, "macrobend", start, dur))
        elif roll < 0.24:
            start = int(RNG.integers(50, N_TIMESTEPS - 80))
            dur = int(RNG.integers(30, 90))
            power = _inject_eavesdrop_tap(power, start, dur)
            label[start:start + dur] = 2
            event_log.append((onu_id, "eavesdrop_tap", start, dur))
        elif roll < 0.35:
            start = int(RNG.integers(50, N_TIMESTEPS - 80))
            dur = int(RNG.integers(10, 30))
            temp, volt = _inject_brute_force(temp, volt, start, dur)
            label[start:start + dur] = 3
            event_log.append((onu_id, "brute_force", start, dur))

        for t in range(N_TIMESTEPS):
            rows.append({
                "onu_id": onu_id,
                "t": t,
                "power_dbm": round(power[t], 3),
                "temp_c": round(temp[t], 3),
                "volt": round(volt[t], 4),
                "label": int(label[t]),
            })

    df = pd.DataFrame(rows)
    log_df = pd.DataFrame(event_log, columns=["onu_id", "event_type", "start", "duration"])
    return df, log_df


if __name__ == "__main__":
    df, log_df = generate_dataset()
    df.to_csv("data/telemetry.csv", index=False)
    log_df.to_csv("data/event_log.csv", index=False)
    print(f"Generated {len(df)} telemetry rows across {N_ONUS} ONUs")
    print(f"Injected {len(log_df)} events:")
    print(log_df["event_type"].value_counts())
