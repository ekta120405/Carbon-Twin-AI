"""
src/layer1_data/splits.py

Layer 1.7: leakage-safe splitting protocol (Overview v2 SS4.8).

NEVER random-shuffle 15-minute solar data -- adjacent timestamps are highly
autocorrelated, so a shuffled split leaks information from t+15min sitting
in train while t sits in test. Split by CONTIGUOUS DAY BLOCKS instead.
"""
from __future__ import annotations

import pandas as pd



def assign_day_blocks(
    index: pd.DatetimeIndex,
    train_days: tuple[int, int] = (1, 20),
    calib_days: tuple[int, int] = (21, 25),
    test_clean_days: tuple[int, int] = (26, 30),
    test_injected_days: tuple[int, int] = (31, 34),
) -> pd.Series:
    """
    Returns a Series aligned to `index`, with values in
    {"train", "calib", "test_clean", "test_injected", "out_of_range"}.

    Day numbering is 1-indexed from the first date present in the data
    (the padding days added in acquisition.py for downscaling context fall
    outside all four named blocks and are tagged "out_of_range" -- they
    exist only to give the CSI interpolation edge context, not for training).
    """
    first_day = index.normalize().min()
    day_number = (index.normalize() - first_day).days + 1  # 1-indexed

    def block_of(d: int) -> str:
        if train_days[0] <= d <= train_days[1]:
            return "train"
        if calib_days[0] <= d <= calib_days[1]:
            return "calib"
        if test_clean_days[0] <= d <= test_clean_days[1]:
            return "test_clean"
        if test_injected_days[0] <= d <= test_injected_days[1]:
            return "test_injected"
        return "out_of_range"

    return pd.Series([block_of(d) for d in day_number], index=index, name="split")


def assert_splits_disjoint(*day_ranges: tuple[int, int]) -> None:
    """Leakage guard: no two named blocks may share a day. Same check as
    tests/test_smoke.py, exposed here so the pipeline can call it at runtime."""
    seen: set[int] = set()
    for lo, hi in day_ranges:
        block = set(range(lo, hi + 1))
        overlap = block & seen
        assert not overlap, f"split blocks overlap on day(s) {overlap} -- leakage risk"
        seen |= block
