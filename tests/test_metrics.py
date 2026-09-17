"""Scoring/label-mapping tests against hand-computed values."""
from conftest import record

from jevbench import metrics


def test_accuracy_recipe_matches_hand_computation():
    # magentic-one (multi-agent): 2 traces, 1 agent-correct, 2 step-correct, modes V.1/V.2 both correct
    # smolagents (single-agent): 2 traces, agent excluded from Who, 1 step-correct, 1 mode-correct
    records = [
        record("magentic-one", agent=True, step=True, mode=True, gt_mode="V.1", pr_mode="V.1"),
        record("magentic-one", agent=False, step=True, mode=True, gt_mode="V.2", pr_mode="V.2"),
        record("smolagents", agent=True, step=True, mode=True, gt_mode="A.3", pr_mode="A.3"),
        record("smolagents", agent=False, step=False, mode=False, gt_mode="R.2", pr_mode="A.1"),
    ]
    acc = metrics.accuracy(records)
    # Who = mean over multi-agent frameworks only = magentic-one agent acc = 1/2
    assert acc["Who"] == 0.5
    # When = mean over frameworks of step acc = (magentic 2/2 + smol 1/2)/2 = 0.75
    assert acc["When"] == 0.75
    # All = mean over frameworks of joint = (magentic 1/2 + smol 1/2)/2 = 0.5
    assert acc["All"] == 0.5
    # What = macro-F1 over modes with gold support: V.1,V.2,A.3 perfect (F1=1), R.2 gold missed (F1=0)
    # classes with gold support: V.1,V.2,A.3,R.2 -> (1+1+1+0)/4 = 0.75
    assert abs(acc["What"] - 0.75) < 1e-9


def test_error_records_excluded():
    records = [
        record("magentic-one", agent=True, step=True, mode=True, gt_mode="V.1", pr_mode="V.1"),
        record("magentic-one", agent=True, step=True, mode=True, gt_mode="V.1", pr_mode="V.1",
               error="RateLimitError: 429"),
    ]
    assert metrics.accuracy(records)["When"] == 1.0  # errored trace dropped
    assert metrics.usage_totals(records)["input_tokens"] == 200  # totals still count both


def test_breakdown_and_latency():
    records = [
        record("smolagents", agent=True, step=True, mode=True, gt_mode="A.3", pr_mode="A.3",
               latency=0.5),
        record("smolagents", agent=False, step=False, mode=False, gt_mode="R.2", pr_mode="A.1",
               latency=2.5),
    ]
    bd = metrics.breakdown(records, "framework")["smolagents"]
    assert bd["n"] == 2 and bd["step"] == 0.5 and bd["joint"] == 0.5
    lat = metrics.latency_stats(records)
    assert lat["median"] == 1.5 and lat["n"] == 2


def test_calibration_ece():
    # Two confident-correct, one overconfident-wrong on the mode axis.
    records = [
        record("magentic-one", agent=True, step=True, mode=True, gt_mode="V.1", pr_mode="V.1",
               probs={"mode": {"V.1": 0.9}}),
        record("magentic-one", agent=True, step=True, mode=True, gt_mode="V.1", pr_mode="V.1",
               probs={"mode": {"V.1": 0.9}}),
        record("magentic-one", agent=True, step=True, mode=False, gt_mode="V.2", pr_mode="V.1",
               probs={"mode": {"V.1": 0.8}}),
    ]
    cal = metrics.calibration(records, "mode")
    assert cal is not None and cal["n"] == 3 and 0.0 <= cal["ece"] <= 1.0


def test_calibration_none_without_probs():
    records = [record("smolagents", agent=True, step=True, mode=True, gt_mode="A.3", pr_mode="A.3")]
    assert metrics.calibration(records, "mode") is None


def test_bootstrap_ci_brackets_point_estimate():
    records = [
        record("magentic-one", agent=(i % 2 == 0), step=True, mode=True,
               gt_mode="V.1", pr_mode="V.1") for i in range(20)
    ]
    ci = metrics.bootstrap_ci(records, n_boot=200, seed=0)
    lo, hi = ci["When"]
    assert lo <= 1.0 <= hi  # step always correct here
