# CarbonTwinAI

A digital-twin framework for verifying solar carbon-credit claims against data the audited plant cannot influence.

## Overview

Solar carbon credits are usually verified with generation data that the plant reports about itself. That leaves room for meter tampering, sensor manipulation, and fabricated records. CarbonTwinAI checks a plant's self-reported output against an independent estimate built only from NASA POWER weather data.

The framework has four layers:

1. **Ingestion and feature engineering.** Aligns the plant stream and the NASA POWER stream, and keeps weather-only "twin" features separate from on-site "evidence" features.
2. **Physics-anchored digital twin.** A physics-based clear-sky anchor (PVWatts) plus a monotonicity-constrained LightGBM quantile-regression model estimates expected AC output.
3. **Calibrated anomaly detection.** A one-sided split-conformal test on daily residual energy, deterministic night-injection and stuck-sensor guardrails, CUSUM drift tracking, an autoencoder, and a gated secondary-evidence escalation step for borderline days.
4. **Explainability and carbon accounting.** Two-tier SHAP attribution for every flag, with excess energy priced in tCO2e and monetary value at risk.

Legitimate operational faults (outages, soiling, curtailment) are kept separate from over-claims, so an auditor can tell a maintenance problem from fraud.

**Data:** the Kaggle dataset `anikannal/solar-power-generation-data` (two plants, 34 days, May to June 2020) and NASA POWER hourly weather data.

## Main Result

Evaluated on a frozen fraud-injection benchmark covering 8 fraud types and 7 legitimate edge cases across two real plants (540 day-instances):

| Metric | Value |
|---|---|
| Precision | 0.817 |
| Recall | 0.648 |
| F1 | 0.723 |
| AUC-PR | 0.766 |
| Specificity on legitimate cases | 0.920 |

- Night-time fabrication and frozen-sensor tampering are detected in every test instance.
- Secondary-evidence escalation raised recall from 58.5% to 64.8% with no loss in specificity.
- Conformal calibration beat a fixed two-sigma threshold on recall, F1, and AUC-PR at identical specificity.
- Detection works at the daily level. Single-slot or hourly overstatements went undetected up to 30% magnitude.

This is a validated proof of concept. The main limitation is dataset size: 34 days, two plants, one season.

## Team

| Name | GitHub |
|---|---|
| Ekta Baphana | [@ekta120405](https://github.com/ekta120405) |
| Ishitaa Jain | [@IshitaaJain24](https://github.com/IshitaaJain24) |
| Aayra Charania | [@aayracharania](https://github.com/aayracharania) |
| Nishi Chhajed | [@nishiiic](https://github.com/nishiiic) |
