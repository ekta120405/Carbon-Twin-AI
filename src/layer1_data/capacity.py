"""
src/layer1_data/capacity.py

Layer 1.6: nameplate capacity estimation (Overview v2 SS4.7).

Nameplate AC capacity isn't published for these plants. Estimate it as the
99.5th percentile of plant-level summed AC power during high-clear-sky-index
conditions -- i.e. "what did this plant actually produce on its best,
clearest days", which is the only capacity proxy available. FREEZE this
constant once estimated; every Layer 3 normalised-residual metric depends
on it staying fixed for the rest of the project.
"""
from __future__ import annotations

import pandas as pd


def estimate_ac_nameplate(
    plant_power_kw: pd.Series,
    clearsky_index: pd.Series,
    percentile: float = 99.5,
    min_kt: float = 0.90,
) -> float:
    """
    Returns a single float: the frozen nameplate capacity estimate (kW).
    """
    aligned = pd.DataFrame({"power": plant_power_kw, "kt": clearsky_index}).dropna()
    clear = aligned[aligned["kt"] >= min_kt]
    if len(clear) < 20:
        raise ValueError(
            f"Only {len(clear)} clear-sky (Kt>={min_kt}) samples available -- "
            f"too few for a stable capacity estimate. Lower min_kt or check "
            f"the Kt computation upstream."
        )
    return float(clear["power"].quantile(percentile / 100.0))
