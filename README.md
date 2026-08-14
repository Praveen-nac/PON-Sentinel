# PON-Sentinel

**Explainable, ML-based eavesdropping & fiber-tap detection for GPON/EPON access networks.**

A natural extension of [GPON-Guard](https://github.com/Praveen-nac/GPON-Guard-Statistical-Anomaly-Detection-System-for-GPON-EPON-Networks): where GPON-Guard used rule-based checks and rolling z-score baselines to catch physical-layer anomalies, PON-Sentinel replaces the single-threshold approach with an **unsupervised multivariate ML model (Isolation Forest)** and adds an **explainability layer (SHAP)** so every flagged event comes with a human-readable reason — not just a black-box alarm.

## Why this project exists

Passive Optical Networks (PON/GPON/EPON) broadcast downstream traffic to every ONU on a shared fiber tree and rely on a shared upstream channel. That structure is efficient, but it also means a physically inserted optical coupler (a **fiber tap**) can silently siphon a small, stable fraction of signal — the kind of subtle, low-variance power drop that a simple threshold rule tends to miss or confuse with normal signal drift.

Meanwhile, most cybersecurity research and tooling focuses on the software/cloud layer. The physical access layer — the OLT/ONU/fiber links that actually carry the traffic — gets comparatively little attention, despite being exposed to tampering, eavesdropping, and denial-of-service in ways software-layer tools can't see. Recent research (2026) on optical network security and PON eavesdropping detection points at exactly this gap: ML-based detection of subtle physical-layer signatures, with a growing emphasis on **explainability** so operators can trust and act on what the model flags.

This project is a hands-on exploration of that gap, building on 3+ years of live GPON/EPON field experience running [Praveen Broadband](https://github.com/Praveen-nac).

## What it does

1. **Simulates realistic PON telemetry** for 32 ONUs (optical power, temperature, voltage) with three distinct injected event types, each with a deliberately different signature:
   - **Macrobend / benign fault** — larger power drop, high variance, partial recovery
   - **Eavesdropping / fiber tap** — small, sudden, *persistent* power drop with unusually *low* variance (the signature a naive threshold rule struggles to separate from normal noise)
   - **OLT brute-force** — correlated temperature/voltage jitter on the management interface

2. **Engineers rolling-window features** (mean, std, deltas, deviation-from-baseline) that expose the shape of each signature, not just its raw value.

3. **Trains an unsupervised Isolation Forest** on known-normal data only — mirroring real deployment, where labeled attack data essentially doesn't exist and a system has to calibrate on a healthy baseline period.

4. **Explains every flagged point with SHAP**, surfacing the top contributing features per detection — so "ONU #12 flagged at t=356, mainly due to `power_dev_from_baseline` and low `power_roll_std`" is distinguishable from a bend, at a glance.

5. **Serves results through a live dashboard** (Flask + Chart.js) — per-ONU telemetry plots with anomaly points overlaid, plus a reasons panel.

## Results (on synthetic data)

| Event type | Detection rate |
|---|---|
| Macrobend (benign fault) | 100% |
| **Eavesdropping / fiber tap** | **95.7%** |
| OLT brute-force | 24%* |
| Normal (false-positive rate) | 5.0% |

*Brute-force detection is weaker because it's deliberately modeled as a subtle secondary signal (temp/voltage jitter) rather than a direct optical-layer effect — a known limitation, noted below as future work.

## Project structure

```
├── simulate.py      # synthetic telemetry generator with 3 injected attack/fault types
├── features.py       # rolling-window feature engineering
├── detector.py        # Isolation Forest training + scoring
├── explain.py           # SHAP explainability layer
├── app.py                 # Flask API + dashboard
├── templates/index.html    # dashboard UI
├── data/                     # generated CSVs (telemetry, features, scored results)
└── models/                     # saved model + scaler
```

## Running it

```bash
pip install -r requirements.txt

python simulate.py      # generates data/telemetry.csv
python features.py       # generates data/features.csv
python detector.py        # trains model, prints per-event detection rates
python explain.py          # SHAP explanations for a sample of flagged points

python app.py               # dashboard at http://localhost:5050
```

## Relationship to GPON-Guard

| | GPON-Guard (v1) | PON-Sentinel (this project) |
|---|---|---|
| Detection method | Rule-based + single-feature z-score | Unsupervised multivariate ML (Isolation Forest) |
| Attack coverage | Rogue ONU, tampering, brute-force, bandwidth theft | Adds a dedicated eavesdropping/fiber-tap signature |
| Trust / transparency | Threshold triggers, no "why" | SHAP-based per-detection explanation |
| Interface | Live dashboard + alerting | Live dashboard + per-ONU explainability panel |

## Limitations (honest notes, not glossed over)

This is a proof-of-concept, not a validated research result — flagging these openly rather than overselling them:

- **Synthetic data only.** The attack/fault signatures (tap = small + persistent + low-variance, bend = larger + high-variance) were designed by hand based on optical networking principles, not measured from a real fiber tap. The model is therefore learning to detect the signature I told it to expect — real-world taps may not be this clean, and validating against real GPON OLT telemetry is the necessary next step before trusting these numbers.
- **Small event count.** The 95.7% eavesdrop detection rate comes from only 2 injected tap events (116 timesteps total) out of 32 simulated ONUs — encouraging as a signal, but not statistically strong enough to generalize from yet. More simulated runs (or real data) are needed before this is a defensible research claim.
- **No baseline comparison yet.** I haven't run GPON-Guard v1's threshold-based rule on this same dataset to quantify how much the ML + explainability approach actually improves over it — that comparison is missing and would strengthen the case considerably.
- **Brute-force detection is weak (24%)** because it's modeled as an indirect signal (temperature/voltage jitter) rather than actual OLT console log data, which wasn't available in simulation.

## Future work

- Improve brute-force recall by incorporating OLT console log timing directly rather than proxying through temperature/voltage jitter
- Move from synthetic telemetry to real GPON OLT SNMP/OMCI counters where accessible
- Explore autoencoder-based reconstruction error as an alternative to Isolation Forest, and compare explainability trade-offs
- Extend to multi-OLT correlation — detecting coordinated tap patterns across a PON tree rather than per-ONU in isolation

---

Built by [Praveen Kumar Gupta](https://github.com/Praveen-nac) — network engineer, 3+ years running GPON/EPON infrastructure at Praveen Broadband, exploring physical-layer telecom security ahead of MEXT-funded graduate research in Japan.
