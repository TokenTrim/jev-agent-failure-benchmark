"""Compare a Jev run against the published Who&When Pro text-subset numbers.

The baseline rows are the paper's Table 4 (arXiv:2607.09996), text modality. Our
Jev row is scored with the official recipe on the same subset. Who and When are
labelled adaptation-favoured because Jev picks from enumerated candidates while
the paper's LLMs free-generate; What (error macro-F1) is the like-for-like axis.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics

# arXiv:2607.09996, Table 4, TEXT modality. Percentages.
PAPER_TEXT = {
    "GPT-5.4": {"Who": 55.7, "When": 72.3, "What": 15.3, "All": 21.3},
    "Claude Sonnet 4.6": {"Who": 54.9, "When": 69.8, "What": 19.1, "All": 22.4},
    "GLM-5": {"Who": 54.9, "When": 71.1, "What": 22.2, "All": 25.3},
    "Qwen3.5-122B": {"Who": 57.5, "When": 73.9, "What": 17.0, "All": 21.6},
}
AXES = ("Who", "When", "What", "All")
JEV_PRICE_PER_MTOK = 0.042  # input only; output free


def jev_row(results_path: Path) -> dict:
    recs = metrics.load_records(results_path)
    acc = metrics.accuracy(recs)
    ci = metrics.bootstrap_ci(recs, n_boot=1000, seed=0)
    intok = metrics.usage_totals(recs)["input_tokens"]
    cal = metrics.calibration(recs, "mode")
    return {
        "n": sum(1 for r in recs if not r.get("error")),
        "acc": {ax: (acc[ax] * 100 if acc[ax] is not None else None) for ax in AXES},
        "ci": {ax: ([c * 100 for c in ci[ax]] if ci[ax] else None) for ax in AXES},
        "input_tokens": intok,
        "cost_usd": round(intok / 1e6 * JEV_PRICE_PER_MTOK, 2),
        "mode_ece": round(cal["ece"], 3) if cal else None,
    }


def render_markdown(jev: dict, *, label: str = "Jev") -> str:
    best = {ax: max(PAPER_TEXT[m][ax] for m in PAPER_TEXT) for ax in AXES}
    L = ["# Who&When Pro — Jev vs frontier LLMs (text subset)", ""]
    L.append("| Model | Who | When | What | All |")
    L.append("|---|---|---|---|---|")

    def jc(ax: str) -> str:
        v = jev["acc"][ax]
        if v is None:
            return "n/a"
        ci = jev["ci"][ax]
        s = f"**{v:.1f}**"
        return s + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")

    L.append(f"| {label} (ours, n={jev['n']}) | {jc('Who')} | {jc('When')} | {jc('What')} | {jc('All')} |")
    for m, r in PAPER_TEXT.items():
        L.append(f"| {m} *(paper)* | {r['Who']:.1f} | {r['When']:.1f} | {r['What']:.1f} | {r['All']:.1f} |")

    L += [
        "",
        (f"Jev cost: **${jev['cost_usd']:.2f}** for all {jev['n']} traces "
         f"({jev['input_tokens']:,} input tokens at ${JEV_PRICE_PER_MTOK}/Mtok; output free). "
         "Latency is not compared here (the endpoint is a shared early-access service; use Jev's "
         "published per-token throughput instead of our queue-affected timings)."),
        "",
        ("**Read this carefully.** Who and When are **adaptation-favoured**: Jev's typed API is "
         "handed the trace's enumerated agent ids and step coordinates and picks one, while the "
         "paper's LLMs free-generate the answer. The like-for-like axis is **What** (error "
         "macro-F1) where both sides classify over the identical 17-code taxonomy: "
         f"Jev {jev['acc']['What']:.1f} vs the best LLM {best['What']:.1f}. "
         f"Mode-confidence calibration ECE: {jev['mode_ece']}. "
         "Baselines: arXiv:2607.09996 Table 4. Not a leaderboard submission."),
    ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--label", default="Jev")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    row = jev_row(a.results)
    md = render_markdown(row, label=a.label)
    print(md)
    print(json.dumps(row, indent=2))
    if a.out:
        a.out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
