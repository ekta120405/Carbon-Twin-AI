"""
src/layer2_twin/build_comparison_table.py

Compiles physics-only, PI-GBQR twin, Ridge, and LSTM metrics (already
computed and saved by train_twin.py and compare_baselines.py) into one
comparison table per plant, for the paper's benchmark section.

Usage:
    python -m src.layer2_twin.build_comparison_table
"""
import json
from pathlib import Path


def main():
    for plant_key in ("plant_1", "plant_2"):
        with open(f"reports/tables/{plant_key}_twin_metrics.json") as fh:
            twin_data = json.load(fh)
        with open(f"reports/tables/{plant_key}_baseline_metrics.json") as fh:
            baseline_data = json.load(fh)

        print(f"\n{'='*90}")
        print(f"{plant_key}  (calibration ratio: {twin_data['calibration_ratio']:.4f})")
        print(f"{'='*90}")
        print(f"{'regime':22s} {'physics-only R2':>16s} {'PI-GBQR R2':>12s} {'Ridge R2':>10s} {'LSTM R2':>10s}")

        regimes = list(twin_data["physics_only"].keys())
        for regime in regimes:
            phys_r2 = twin_data["physics_only"][regime].get("r2")
            twin_r2 = twin_data["pigbqr"][regime].get("r2")
            ridge_r2 = baseline_data["ridge"][regime].get("r2")
            lstm_r2 = baseline_data["lstm"][regime].get("r2") if baseline_data["lstm"] else None

            def fmt(x):
                return f"{x:.3f}" if x is not None else "n/a"

            print(f"{regime:22s} {fmt(phys_r2):>16s} {fmt(twin_r2):>12s} {fmt(ridge_r2):>10s} {fmt(lstm_r2):>10s}")

    out_path = Path("reports/tables/model_comparison_summary.md")
    with open(out_path, "w") as fh:
        fh.write("# Layer 2 Model Comparison (R^2 by regime, test_clean)\n\n")
        for plant_key in ("plant_1", "plant_2"):
            with open(f"reports/tables/{plant_key}_twin_metrics.json") as fh2:
                twin_data = json.load(fh2)
            with open(f"reports/tables/{plant_key}_baseline_metrics.json") as fh2:
                baseline_data = json.load(fh2)
            fh.write(f"\n## {plant_key} (CR={twin_data['calibration_ratio']:.4f})\n\n")
            fh.write("| regime | physics-only | PI-GBQR | Ridge | LSTM |\n")
            fh.write("|---|---|---|---|---|\n")
            for regime in twin_data["physics_only"].keys():
                phys_r2 = twin_data["physics_only"][regime].get("r2")
                twin_r2 = twin_data["pigbqr"][regime].get("r2")
                ridge_r2 = baseline_data["ridge"][regime].get("r2")
                lstm_r2 = baseline_data["lstm"][regime].get("r2") if baseline_data["lstm"] else None
                def fmt(x):
                    return f"{x:.3f}" if x is not None else "n/a"
                fh.write(f"| {regime} | {fmt(phys_r2)} | {fmt(twin_r2)} | {fmt(ridge_r2)} | {fmt(lstm_r2)} |\n")
    print(f"\nSaved markdown summary to {out_path}")


if __name__ == "__main__":
    main()
