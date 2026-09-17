"""Assemble the head-to-head comparison from per-backend record files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import metrics
from .pricing import cost_usd, price_for

AXES = ("Who", "When", "What", "All")


def backend_summary(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Accuracy, CIs, latency, tokens and estimated cost for one backend."""
    usage = metrics.usage_totals(records)
    cost = cost_usd(model, usage["input_tokens"], usage["output_tokens"])
    n_ok = sum(1 for r in records if not r.get("error"))
    return {
        "model": model,
        "n": n_ok,
        "n_error": sum(1 for r in records if r.get("error")),
        "accuracy": metrics.accuracy(records),
        "ci95": metrics.bootstrap_ci(records),
        "latency_s": metrics.latency_stats(records),
        "usage": usage,
        "cost_usd": cost,
        "priced": price_for(model) is not None,
    }


def build_report(run_dir: Path, backends: dict[str, str]) -> dict[str, Any]:
    """Compare each backend under `run_dir`. `backends` maps name -> model id."""
    out: dict[str, Any] = {"backends": {}, "breakdown_framework": {}, "calibration": {}}
    for name, model in backends.items():
        records = metrics.load_records(run_dir / name / "text.jsonl")
        out["backends"][name] = backend_summary(records, model)
        out["breakdown_framework"][name] = metrics.breakdown(records, "framework")
        cal = {ax: metrics.calibration(records, ax) for ax in ("agent", "step", "mode")}
        out["calibration"][name] = {k: v for k, v in cal.items() if v is not None}
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    """A compact results report. Honest about protocol deviations and gaps."""
    lines = ["# Results", "", "## Accuracy (official Who&When Pro recipe, text subset)", ""]
    names = list(report["backends"])
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for ax in AXES:
        row = [ax]
        for n in names:
            acc = report["backends"][n]["accuracy"].get(ax)
            ci = (report["backends"][n]["ci95"] or {}).get(ax)
            cell = _fmt(acc)
            if ci:
                cell += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Latency, tokens, cost", ""]
    lines.append("| Backend | Model | median s | p95 s | input tok | output tok | est. cost |")
    lines.append("|---|---|---|---|---|---|---|")
    for n in names:
        b = report["backends"][n]
        lat, u = b["latency_s"], b["usage"]
        cost = "n/a" if b["cost_usd"] is None else f"${b['cost_usd']:.2f}"
        cost += "" if b["priced"] else " (unpriced)"
        lines.append(
            f"| {n} | {b['model']} | {lat['median']:.2f} | {lat['p95']:.2f} | "
            f"{u['input_tokens']} | {u['output_tokens']} | {cost} |"
        )
    lines.append("")
    lines.append("Costs are estimates from published rates, not billed amounts. "
                 "Who counts multi-agent frameworks only; What is macro-F1 over error modes. "
                 "Who&When Pro failures are injected by a controlled pipeline, not natural incidents.")
    return "\n".join(lines) + "\n"


def render_chart(report: dict[str, Any], path: Path) -> None:
    """Grouped-bar accuracy chart plus a latency annotation. Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(report["backends"])
    x = np.arange(len(AXES))
    width = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, n in enumerate(names):
        vals = [report["backends"][n]["accuracy"].get(a) or 0.0 for a in AXES]
        ax.bar(x + i * width, vals, width, label=f"{n} ({report['backends'][n]['model']})")
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(AXES)
    ax.set_ylabel("accuracy / macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Who&When Pro (text subset): failure attribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(run_dir: Path, backends: dict[str, str]) -> dict[str, Any]:
    """Build the report, write results.json and REPORT.md into `run_dir`."""
    report = build_report(run_dir, backends)
    (run_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report
