"""Score a Jev run and compare it to the published Who&When Pro text numbers.

Baselines are the paper's Table 4 (arXiv:2607.09996), text modality. Our Jev row
is scored with the official recipe on the same subset. Who and When are
adaptation-favoured (Jev picks from enumerated candidates; the LLMs free-generate);
What (error macro-F1) is the like-for-like axis. Writes RESULTS.md and a chart.
"""
from __future__ import annotations

import argparse
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
CONSTRAINED = {"Who", "When"}  # adaptation-favoured for Jev
JEV_PRICE_PER_MTOK = 0.042  # input only; output free


def jev_row(results_path: Path) -> dict:
    """Official Who/When/What/All (as percentages) plus cost and calibration."""
    recs = metrics.load_records(results_path)
    acc = metrics.accuracy(recs)
    ci = metrics.bootstrap_ci(recs, n_boot=1000, seed=0)
    intok = metrics.usage_totals(recs)["input_tokens"]
    cal = metrics.calibration(recs, "mode")
    return {
        "n": sum(1 for r in recs if not r.get("error")),
        "acc": {ax: acc[ax] * 100 if acc[ax] is not None else None for ax in AXES},
        "ci": {ax: [c * 100 for c in ci[ax]] if ci[ax] else None for ax in AXES},
        "input_tokens": intok,
        "cost_usd": round(intok / 1e6 * JEV_PRICE_PER_MTOK, 2),
        "mode_ece": round(cal["ece"], 3) if cal else None,
    }


def _best_llm() -> dict[str, tuple[str, float]]:
    """Per axis, the best paper model and its score."""
    return {ax: max(((m, PAPER_TEXT[m][ax]) for m in PAPER_TEXT), key=lambda t: t[1])
            for ax in AXES}


def render_markdown(jev: dict, *, label: str = "Jev") -> str:
    best = _best_llm()
    L = ["# Who&When Pro - Jev vs frontier LLMs (text subset)", "",
         "| Model | Who | When | What | All |", "|---|---|---|---|---|"]

    def cell(ax: str) -> str:
        v, ci = jev["acc"][ax], jev["ci"][ax]
        return "n/a" if v is None else f"**{v:.1f}**" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")

    L.append(f"| {label} (ours, n={jev['n']}) | {cell('Who')} | {cell('When')} | "
             f"{cell('What')} | {cell('All')} |")
    for m, r in PAPER_TEXT.items():
        L.append(f"| {m} *(paper)* | {r['Who']:.1f} | {r['When']:.1f} | {r['What']:.1f} | {r['All']:.1f} |")

    L += [
        "",
        (f"Jev cost: **${jev['cost_usd']:.2f}** for all {jev['n']} traces "
         f"({jev['input_tokens']:,} input tokens at ${JEV_PRICE_PER_MTOK}/Mtok; output free)."),
        "",
        ("**Who and When are adaptation-favoured**: Jev picks the agent/step from the trace's "
         "enumerated options, while the paper's LLMs free-generate. The like-for-like axis is "
         f"**What** (error macro-F1, same 17-code taxonomy for both): Jev {jev['acc']['What']:.1f} "
         f"vs the best LLM {best['What'][1]:.1f}. Mode-confidence ECE {jev['mode_ece']}. "
         "Latency is not compared (shared early-access endpoint). Baselines: arXiv:2607.09996 "
         "Table 4. Failures in Who&When Pro are injected, not natural incidents."),
    ]
    return "\n".join(L) + "\n"


def render_chart(jev: dict, path: Path) -> None:
    """Grouped bars: Jev vs the best paper LLM per axis. Who/When hatched."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = _best_llm()
    blue, orange, surface, ink, muted = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    w = 0.38
    for i, k in enumerate(AXES):
        jx, lx = i - w / 2, i + w / 2
        ax.bar(jx, jev["acc"][k], w, color=blue, edgecolor=surface, linewidth=1.5,
               hatch="///" if k in CONSTRAINED else None, zorder=3,
               label="Jev (ours)" if i == 0 else None)
        ax.bar(lx, best[k][1], w, color=orange, edgecolor=surface, linewidth=1.5, zorder=3,
               label="Best frontier LLM (paper)" if i == 0 else None)
        ax.text(jx, jev["acc"][k] + 1.2, f"{jev['acc'][k]:.1f}", ha="center", va="bottom",
                fontsize=10.5, color=ink, fontweight="bold")
        ax.text(lx, best[k][1] + 1.2, f"{best[k][1]:.1f}", ha="center", va="bottom",
                fontsize=10.5, color=muted)
        ax.text(lx, 1.5, best[k][0], ha="center", va="bottom", fontsize=7, color=muted)

    labels = {"Who": "Who\n(agent)", "When": "When\n(step)", "What": "What\n(error F1)",
              "All": "All\n(joint)"}
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels([labels[k] for k in AXES], fontsize=10, color=ink)
    ax.set_ylim(0, 90)
    ax.set_ylabel("accuracy / macro-F1  (%)", fontsize=10, color=muted)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(length=0, colors=muted)
    ax.yaxis.grid(True, color="#ecebe7", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Finding what broke an AI agent: Jev vs frontier LLMs", fontsize=15,
                 color=ink, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.02, "Who&When Pro, text subset (6,257 traces). Higher is better.",
            transform=ax.transAxes, fontsize=10, color=muted)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.text(0.075, 0.055,
             (f"Jev ran all {jev['n']:,} traces for ~${jev['cost_usd']:.2f} (input-only; output "
              "free).  Hatched = constrained-choice for Jev: it picks from the trace's listed "
              "agents/steps,"), fontsize=7.2, color=muted)
    fig.text(0.075, 0.018,
             ("while the LLMs free-generate. What (error macro-F1) is the like-for-like axis.  "
              "Baselines: best of 4 frontier LLMs per axis, arXiv:2607.09996 Table 4."),
             fontsize=7.2, color=muted)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.19)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--label", default="Jev")
    p.add_argument("--out", type=Path, default=Path("RESULTS.md"))
    p.add_argument("--chart", type=Path, default=Path("figures/whowhen_jev_vs_llm.png"))
    a = p.parse_args(argv)
    row = jev_row(a.results)
    a.out.write_text(render_markdown(row, label=a.label), encoding="utf-8")
    render_chart(row, a.chart)
    print(render_markdown(row, label=a.label))
    print(json.dumps(row))
    print(f"wrote {a.out} and {a.chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
