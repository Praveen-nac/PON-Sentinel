# PON-Sentinel

ML-based eavesdropping and fiber-tap detection for GPON/EPON access networks, with explainable alerts.

This is a follow-up to [GPON-Guard](https://github.com/Praveen-nac/GPON-Guard-Statistical-Anomaly-Detection-System-for-GPON-EPON-Networks), where I used rule-based checks and rolling z-score baselines to catch physical-layer anomalies. That approach works but a single threshold can't really separate a fiber tap from normal signal drift — they can look similar on one metric alone. So here I swapped it out for an unsupervised Isolation Forest that looks at multiple features together, and added SHAP so every alert comes with an actual reason instead of just a number crossing a line.

## Why I built this

PON/GPON/EPON networks broadcast downstream traffic to every ONU on a shared fiber tree, and the upstream is shared too. That's efficient, but it also means someone can physically splice in an optical coupler and quietly skim off a small, stable slice of the signal. That kind of tap tends to show up as a small, persistent power drop with unusually low variance — exactly the sort of subtle pattern a simple threshold rule tends to either miss or flag as noise.

Most security tooling I've come across looks at the software/cloud side of things. The physical layer — the actual OLT, ONUs, and fiber that carry the traffic — doesn't get nearly as much attention, even though it's just as exposed to tampering and eavesdropping. I've been running GPON/EPON infrastructure hands-on for 3+ years at Praveen Broadband, and this project is me digging into that gap with an ML angle instead of hand-tuned rules.

## What it does

- Simulates telemetry (optical power, temperature, voltage) for 32 ONUs, with three injected event types, each with a different signature:
  - **Macrobend / benign fault** — larger power drop, high variance, partial recovery
  - **Eavesdropping / fiber tap** — small, sudden, persistent power drop with unusually low variance (the hard one for a naive threshold to catch)
  - **OLT brute-force** — correlated temperature/voltage jitter on the management interface
- Builds rolling-window features (mean, std, deltas, deviation-from-baseline) so the model sees the *shape* of an event, not just a raw reading
- Trains an Isolation Forest on known-normal data only — this mirrors real deployments, where you basically never have labeled attack data and have to calibrate off a healthy baseline period instead
- Runs every flagged point through SHAP, so instead of just "anomaly detected" you get something like: *ONU #12 flagged at t=356, mainly due to power_dev_from_baseline and low power_roll_std*
- Serves it all through a small Flask + Chart.js dashboard — per-ONU plots with anomalies overlaid, plus a panel explaining each one

## Results (synthetic data)

| Event type | Detection rate |
|---|---|
| Macrobend (benign fault) | 100% |
| Eavesdropping / fiber tap | 95.7% |
| OLT brute-force | 24%* |
| Normal (false-positive rate) | 5.0% |

*Brute-force is weak because I modeled it as an indirect signal (temp/voltage jitter) rather than actual OLT console log data — more on that below.

## Project structure

```
├── simulate.py       # synthetic telemetry generator, 3 injected event types
├── features.py        # rolling-window feature engineering
├── detector.py         # Isolation Forest training + scoring
├── explain.py           # SHAP explainability layer
├── app.py                # Flask API + dashboard
├── templates/index.html   # dashboard UI
├── data/                    # generated CSVs (telemetry, features, scored results)
└── models/                    # saved model + scaler
```

## Running it

```bash
pip install -r requirements.txt

python simulate.py    # generates data/telemetry.csv
python features.py    # generates data/features.csv
python detector.py    # trains the model, prints per-event detection rates
python explain.py     # SHAP explanations for a sample of flagged points

python app.py          # dashboard at http://localhost:5050
```

## GPON-Guard vs. PON-Sentinel

| | GPON-Guard (v1) | PON-Sentinel |
|---|---|---|
| Detection method | Rule-based + single-feature z-score | Unsupervised, multivariate (Isolation Forest) |
| Attack coverage | Rogue ONU, tampering, brute-force, bandwidth theft | Adds a dedicated eavesdropping/fiber-tap signature |
| Transparency | Threshold triggers, no explanation | SHAP-based reasoning per detection |
| Interface | Live dashboard + alerting | Live dashboard + per-ONU explainability panel |

## Limitations

Not glossing over these — this is a proof-of-concept, not a validated result yet:

- **Synthetic data only.** I designed the tap/bend signatures by hand based on optical networking principles, not from a real fiber tap. So the model is learning to detect the signature I told it to expect. Real taps could look messier — validating against real GPON OLT telemetry is the obvious next step.
- **Small event count.** The 95.7% number comes from just 2 injected tap events (116 timesteps) across 32 ONUs. It's a promising signal, but not enough to generalize a real claim from yet.
- **No head-to-head with GPON-Guard v1.** I haven't run the old threshold rule on this same dataset to see how much the ML + SHAP approach actually improves things. That comparison is missing and would make the case a lot stronger.
- **Brute-force detection is weak (24%)** since it's proxied through temperature/voltage jitter rather than real OLT console log data, which I didn't have access to in simulation.

## What's next

- Improve brute-force recall using OLT console log timing directly instead of proxying through temp/voltage jitter
- Move from synthetic telemetry to real GPON OLT SNMP/OMCI counters where I can get access
- Try an autoencoder-based reconstruction-error approach as an alternative to Isolation Forest and compare explainability trade-offs
- Extend to multi-OLT correlation — catching coordinated tap patterns across a PON tree, not just per-ONU

---

Built by [Praveen Kumar Gupta](https://github.com/Praveen-nac) — network engineer running GPON/EPON infrastructure at Praveen Broadband for 3+ years, exploring physical-layer telecom security ahead of MEXT-funded graduate research in Japan.
