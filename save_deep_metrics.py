import json
import pandas as pd
from pathlib import Path

def compute_full_metrics(plant_key, model_name):
    preds = pd.read_parquet(f"reports/benchmarks/{plant_key}_{model_name}_predictions.parquet")
    df = pd.read_parquet(f"data/processed/{plant_key}.parquet")
    clean = df[df["is_clean"]].copy().asfreq("15min")

    actual_aligned = clean["ac_power"].reset_index(drop=True).iloc[-len(preds):]
    actual_aligned.index = preds.index
    kt_aligned = clean["twin_kt"].reset_index(drop=True).iloc[-len(preds):]
    kt_aligned.index = preds.index

    err = preds["expected_q50"] - actual_aligned
    ss_tot_all = ((actual_aligned - actual_aligned.mean())**2).sum()
    r2_overall = 1 - (err**2).sum()/ss_tot_all

    result = {"overall": {"n": len(actual_aligned), "r2": float(r2_overall)}}
    for lo, hi, name in [(0.75, 1.5, "clear"), (0.4, 0.75, "partly_cloudy"), (0.0, 0.4, "overcast")]:
        mask = (kt_aligned >= lo) & (kt_aligned < hi)
        n = int(mask.sum())
        if n > 5:
            sub_err = err[mask]
            sub_actual = actual_aligned[mask]
            ss_tot_r = ((sub_actual - sub_actual.mean())**2).sum()
            r2_r = 1 - (sub_err**2).sum()/ss_tot_r if ss_tot_r > 1e-9 else None
            result[name] = {"n": n, "r2": float(r2_r) if r2_r is not None else None, "low_n_caveat": n < 20}
        else:
            result[name] = {"n": n, "r2": None, "low_n_caveat": True}
    return result

for plant_key in ("plant_1", "plant_2"):
    for model_name in ("tft", "tsmixer"):
        metrics = compute_full_metrics(plant_key, model_name)
        out_path = Path(f"reports/tables/{plant_key}_{model_name}_r2_metrics.json")
        with open(out_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved {out_path}")
