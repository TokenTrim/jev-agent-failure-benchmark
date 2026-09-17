"""Metrics over a backend's result records.

Accuracy uses the official recipe unchanged (`whowhen_eval.leaderboard`): Who
and When and All are per-framework means, What is a global macro-F1, Who counts
multi-agent frameworks only. On top we add latency percentiles, token/cost
totals, cluster-bootstrap confidence intervals, and calibration for a backend
that returns probabilities.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from whowhen_eval.leaderboard import cell_metrics, macro_f1  # noqa: F401 (recipe reuse)

AXES = ("Who", "When", "What", "All")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read records, keeping the last record per trace id (retries win)."""
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        tid = rec.get("trace_id")
        if tid:
            latest[tid] = rec
    return list(latest.values())


def _cell_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape non-error records for `cell_metrics`."""
    rows: list[dict[str, Any]] = []
    for r in records:
        if r.get("error"):
            continue
        sc = r.get("score") or {}
        gt = r.get("ground_truth") or {}
        pr = r.get("prediction") or {}
        rows.append({
            "fw": r.get("framework"),
            "modality": r.get("modality"),
            "a_correct": bool(sc.get("agent")),
            "s_correct": bool(sc.get("step")),
            "m_correct": bool(sc.get("mode")),
            "gt_mode": str(gt.get("mode") or ""),
            "pr_mode": str(pr.get("error_mode") or ""),
        })
    return rows


def accuracy(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Who/When/What/All for the (text) cell."""
    return cell_metrics(_cell_rows(records)) or {ax: None for ax in AXES}


def bootstrap_ci(
    records: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float] | None]:
    """95% CIs on each axis by cluster bootstrap, resampling within framework."""
    rows = _cell_rows(records)
    if not rows:
        return {ax: None for ax in AXES}
    by_fw: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fw[r["fw"]].append(r)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {ax: [] for ax in AXES}
    for _ in range(n_boot):
        resampled: list[dict] = []
        for fw_rows in by_fw.values():
            idx = rng.integers(0, len(fw_rows), len(fw_rows))
            resampled.extend(fw_rows[i] for i in idx)
        m = cell_metrics(resampled) or {}
        for ax in AXES:
            if m.get(ax) is not None:
                draws[ax].append(m[ax])
    out: dict[str, tuple[float, float] | None] = {}
    for ax in AXES:
        vals = draws[ax]
        out[ax] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else None
    return out


def _joint(r: dict[str, Any]) -> bool:
    sc = r.get("score") or {}
    return bool(sc.get("agent") and sc.get("step") and sc.get("mode"))


def breakdown(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Per-`key` (framework or benchmark) accuracy on each axis and joint."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not r.get("error"):
            groups[str(r.get(key))].append(r)
    out: dict[str, dict[str, Any]] = {}
    for g, rs in sorted(groups.items()):
        n = len(rs)
        out[g] = {
            "n": n,
            "agent": sum(bool((r.get("score") or {}).get("agent")) for r in rs) / n,
            "step": sum(bool((r.get("score") or {}).get("step")) for r in rs) / n,
            "mode": sum(bool((r.get("score") or {}).get("mode")) for r in rs) / n,
            "joint": sum(_joint(r) for r in rs) / n,
        }
    return out


def latency_stats(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Per-request latency percentiles over non-error records."""
    lat = [r["latency_s"] for r in records if not r.get("error") and r.get("latency_s") is not None]
    if not lat:
        return {"n": 0, "median": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.array(lat, dtype=float)
    return {
        "n": len(lat),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }


def usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    """Summed input/output tokens over records that reported usage."""
    in_tok = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in records)
    out_tok = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in records)
    return {"input_tokens": int(in_tok), "output_tokens": int(out_tok)}


_AXIS_FIELD = {"agent": "agent_name", "step": "step_coord", "mode": "error_mode"}


def calibration(records: list[dict[str, Any]], axis: str, *, bins: int = 10) -> dict[str, Any] | None:
    """Expected calibration error for one axis (`agent`/`step`/`mode`).

    Uses the probability the backend assigned to the option it chose, against
    whether that axis scored correct. Returns None when the backend recorded no
    probabilities for the axis.
    """
    field = _AXIS_FIELD[axis]
    pts: list[tuple[float, int]] = []
    for r in records:
        if r.get("error"):
            continue
        probs = (r.get("probabilities") or {}).get(axis)
        chosen = (r.get("prediction") or {}).get(field)
        if not isinstance(probs, dict) or chosen is None:
            continue
        p = probs.get(chosen)
        if p is None:
            continue
        correct = int(bool((r.get("score") or {}).get(axis)))
        pts.append((float(p), correct))
    if not pts:
        return None
    ps = np.array([p for p, _ in pts])
    ys = np.array([y for _, y in pts])
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    reliability = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (ps >= lo) & (ps < hi) if i < bins - 1 else (ps >= lo) & (ps <= hi)
        if not mask.any():
            continue
        conf = float(ps[mask].mean())
        acc = float(ys[mask].mean())
        w = int(mask.sum())
        ece += w / len(ps) * abs(acc - conf)
        reliability.append({"bin": [float(lo), float(hi)], "n": w, "confidence": conf, "accuracy": acc})
    return {"ece": ece, "n": len(pts), "reliability": reliability}
