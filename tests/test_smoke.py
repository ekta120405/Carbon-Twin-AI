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
