#!/usr/bin/env bash
# CarbonTwinAI — repository scaffolding.
# Creates the directory tree, .gitignore, and config skeletons from Overview v2 §11.
#
#   bash setup_repo.sh            # scaffold in the current directory
#
# Idempotent: safe to re-run. Existing files are left untouched.

set -euo pipefail

echo "Scaffolding CarbonTwinAI repository..."

# ---------------------------------------------------------------- directories
mkdir -p config
mkdir -p data/raw/kaggle data/raw/nasa_power data/interim data/processed
mkdir -p src/layer1_data src/layer2_twin src/layer3_anomaly src/layer4_xai
mkdir -p src/injection src/evaluation src/utils
mkdir -p dashboard/pages
mkdir -p tests
mkdir -p notebooks
mkdir -p reports/figures reports/tables reports/evidence_bundles reports/benchmarks
mkdir -p paper/figures

# ------------------------------------------------------- package init files
for d in src src/layer1_data src/layer2_twin src/layer3_anomaly src/layer4_xai \
         src/injection src/evaluation src/utils tests; do
  [ -f "$d/__init__.py" ] || touch "$d/__init__.py"
done

# ------------------------------------------------------------ keep empty dirs
for d in data/raw/kaggle data/raw/nasa_power data/interim data/processed \
         reports/figures reports/tables reports/evidence_bundles reports/benchmarks; do
  [ -f "$d/.gitkeep" ] || touch "$d/.gitkeep"
done

# ----------------------------------------------------------------- gitignore
if [ ! -f .gitignore ]; then
cat > .gitignore <<'EOF'
# environments
.venv/
.venv-deep/
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# credentials — NEVER commit
.kaggle/
kaggle.json
*.env
.env

# raw Kaggle data: redownloadable, keep out of git
data/raw/kaggle/*
!data/raw/kaggle/.gitkeep

# intermediate + processed artefacts: regenerable from the pipeline
data/interim/*
!data/interim/.gitkeep
data/processed/*
!data/processed/.gitkeep

# DO commit data/raw/nasa_power/ — it is small (<1 MB), and the API has
# multi-month latency plus known outages. Committing the cache is what makes
# your results reproducible by a reviewer. See Overview v2 §14 (risk register).

# outputs
reports/figures/*
!reports/figures/.gitkeep
reports/evidence_bundles/*
!reports/evidence_bundles/.gitkeep
mlruns/
*.log

# model binaries
*.pkl
*.pt
*.txt.model
!requirements*.txt

# OS
.DS_Store
Thumbs.db
EOF
echo "  created .gitignore"
fi

# ----------------------------------------------------------- config/config.yaml
if [ ! -f config/config.yaml ]; then
cat > config/config.yaml <<'EOF'
# CarbonTwinAI — master configuration
# Every magic number in the pipeline lives here, not in code.

project:
  name: CarbonTwinAI
  random_seed: 42

paths:
  raw_kaggle:  data/raw/kaggle
  raw_nasa:    data/raw/nasa_power
  interim:     data/interim
  processed:   data/processed
  reports:     reports

# ---------- Layer 1: data ----------
layer1:
  timezone:
    source_tz: Asia/Kolkata      # Kaggle timestamps are naive IST
    pipeline_tz: UTC             # everything downstream is tz-aware UTC
    solar_noon_tolerance_min: 90 # abort if generation peak deviates more

  resample:
    target_freq: 15min
    method: clear_sky_index      # NEVER ffill or linear on irradiance directly
    clearsky_model: ineichen

  quality:
    min_solar_elevation_deg: 10.0  # daylight mask
    ceiling_tolerance: 1.05        # 5% headroom on the physical maximum
    stuck_sensor_run_length: 8     # identical daylight values -> sensor fault
    drop_negative_power: true

  capacity:
    estimate_from_data: true
    percentile: 99.5
    min_clearsky_index: 0.90

  splits:                        # contiguous DAY blocks — never shuffle
    train_days:          [1, 20]
    calib_days:          [21, 25]   # clean only; conformal calibration
    test_clean_days:     [26, 30]
    test_injected_days:  [31, 34]

# ---------- Layer 2: twin ----------
layer2:
  target: residual_to_physics    # options: residual_to_physics | direct
  quantiles: [0.1, 0.5, 0.9]
  model:
    num_leaves: 15
    max_depth: 4
    min_data_in_leaf: 30
    learning_rate: 0.03
    n_estimators: 600
    feature_fraction: 0.7
    bagging_fraction: 0.7
    bagging_freq: 1
    lambda_l2: 5.0
  monotone:
    poa_global: 1                # expected power non-decreasing in irradiance
    clearsky_index: 1
    temp_cell: -1
  online:
    mode: rolling_refit          # simpler and safer than incremental boosting
    window_days: 20
    refit_every_days: 1

  physics_anchor:
    gamma_pdc: -0.004            # /degC, typical c-Si temperature coefficient
    inverter_ac_dc_ratio: 1.10
    system_losses: 0.14          # PVWatts default total loss fraction

# ---------- Layer 3: anomaly detection ----------
layer3:
  residual:
    normalise_by: ac_nameplate
    decision_window: daily        # primary decision level
  conformal:
    alpha: 0.05
    one_sided: true               # over-claims only; under-claims -> fault channel
    stratify_by_clearsky_index: true
    csi_bins: [0.0, 0.4, 0.75, 1.05]
  cusum:
    k: 0.25                       # slack, in residual sigma units
    h: 5.0                        # decision interval
  autoencoder:
    hidden: [8, 4, 8]
    epochs: 200
    learning_rate: 0.001

# ---------- Layer 4: XAI, carbon, reporting ----------
layer4:
  shap:
    explain_flagged_only: true    # full-dataset SHAP is unnecessary
    max_flags_explained: 500
  carbon:
    grid_emission_factor_tco2_per_mwh: 0.79   # CEA India — cite the exact edition
    carbon_price_usd_per_tco2: 10.0
  severity_weights:
    magnitude: 0.4
    confidence: 0.2
    persistence: 0.2
    tco2e_at_risk: 0.2
EOF
echo "  created config/config.yaml"
fi

# ---------------------------------------------------------- config/plants.yaml
if [ ! -f config/plants.yaml ]; then
cat > config/plants.yaml <<'EOF'
# Plant metadata.
# The Kaggle dataset publishes NO coordinates — these are ASSUMPTIONS.
# Task 4 resolves them and runs the 3x3 grid-cell sensitivity analysis.
# Whatever you settle on, record it here and cite it as a limitation in the paper.

plant_1:
  kaggle_generation: Plant_1_Generation_Data.csv
  kaggle_weather:    Plant_1_Weather_Sensor_Data.csv
  latitude:  null        # <-- fill in Task 4
  longitude: null        # <-- fill in Task 4
  altitude_m: null
  tilt_deg: null         # unknown; estimated during Task 5
  azimuth_deg: 180       # assume due south (northern hemisphere)
  ac_nameplate_kw: null  # estimated from data, Layer 1.6
  n_inverters: 22

plant_2:
  kaggle_generation: Plant_2_Generation_Data.csv
  kaggle_weather:    Plant_2_Weather_Sensor_Data.csv
  latitude:  null
  longitude: null
  altitude_m: null
  tilt_deg: null
  azimuth_deg: 180
  ac_nameplate_kw: null
  n_inverters: 22
  notes: "DC power magnitudes differ ~10x from Plant 1 — normalise per plant."
EOF
echo "  created config/plants.yaml"
fi

# -------------------------------------------------------- config/anomalies.yaml
if [ ! -f config/anomalies.yaml ]; then
cat > config/anomalies.yaml <<'EOF'
# Synthetic anomaly injection — Overview v2 §8.
# OWNED BY: Nishi. FROZEN before any Layer 3 threshold tuning (git tag injector-v1.0).
# Parameters are SAMPLED from these ranges, never hand-picked per run.

seed: 20260812
inject_fraction_of_days: 0.18

positive:
  A1_constant_overclaim:   {delta_range: [0.02, 0.40]}
  A2_peak_hour_shaving:    {delta_range: [0.05, 0.30], hours_local: [10, 15]}
  A3_night_injection:      {magnitude_frac_of_capacity: [0.02, 0.15]}
  A4_day_swap:             {source: sunny_day, target: cloudy_day}
  A5_weather_spoofing:     {onsite_irr_inflation: [0.10, 0.50]}
  A6_slow_drift:           {final_delta: [0.05, 0.15], ramp_days: [10, 30]}
  A7_capacity_violation:   {overshoot: [1.02, 1.25]}
  A8_flatline:             {run_length_slots: [8, 40]}

hard_negatives:              # must NOT be flagged as fraud
  N1_inverter_outage:   {source: real_plant_1_event}
  N2_soiling_loss:      {loss_per_day: 0.002, days: [10, 34]}
  N3_curtailment:       {cap_frac_of_capacity: [0.5, 0.8]}
  N4_sensor_dropout:    {missing_run_slots: [4, 24]}
  N5_heavy_cloud_day:   {source: real_low_csi_day}
  N6_extreme_heat:      {t2m_offset_c: [4, 8]}

evaluation_sweep:
  magnitudes: [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30]
  windows: [15min, hourly, daily, weekly, monthly]
EOF
echo "  created config/anomalies.yaml"
fi

# ------------------------------------------------------------- pytest scaffold
if [ ! -f tests/test_smoke.py ]; then
cat > tests/test_smoke.py <<'EOF'
"""Smoke tests. Replaced by the full suite in Task 7."""


def test_config_loads():
    import yaml
    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["layer1"]["timezone"]["pipeline_tz"] == "UTC"
    assert cfg["layer3"]["conformal"]["one_sided"] is True


def test_splits_are_disjoint():
    import yaml
    with open("config/config.yaml") as fh:
        splits = yaml.safe_load(fh)["layer1"]["splits"]
    seen: set[int] = set()
    for lo, hi in splits.values():
        block = set(range(lo, hi + 1))
        assert not (block & seen), "split blocks overlap — leakage risk"
        seen |= block
EOF
echo "  created tests/test_smoke.py"
fi

if [ ! -f pytest.ini ]; then
cat > pytest.ini <<'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
EOF
echo "  created pytest.ini"
fi

echo ""
echo "Done. Tree:"
find . -maxdepth 2 -type d -not -path '*/\.*' | sort | sed 's|^\./||'
echo ""
echo "Next:"
echo "  1. python3.11 -m venv .venv && source .venv/bin/activate"
echo "  2. pip install -r requirements.txt"
echo "  3. python verify_env.py"
echo "  4. pytest"
