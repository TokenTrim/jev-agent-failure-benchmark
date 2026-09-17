"""Render the Jev vs frontier-LLM comparison chart (static PNG for sharing).

Two series (Jev vs the best paper LLM per axis), grouped by metric. Who/When are
hatched to mark that Jev answered them as constrained choice; What is the
like-for-like axis. Palette + surfaces from the dataviz reference instance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["Who", "When", "What", "All"]
JEV = {"Who": 73.4, "When": 76.4, "What": 23.7, "All": 31.3}
BEST_LLM = {"Who": 57.5, "When": 73.9, "What": 22.2, "All": 25.3}
LLM_SRC = {"Who": "Qwen3.5-122B", "When": "Qwen3.5-122B", "What": "GLM-5", "All": "GLM-5"}
CONSTRAINED = {"Who", "When"}  # adaptation-favoured for Jev

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(len(AXES))
    w = 0.38

    for i, k in enumerate(AXES):
        jx, lx = i - w / 2, i + w / 2
        hatch = "///" if k in CONSTRAINED else None
        ax.bar(jx, JEV[k], w, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               hatch=hatch, zorder=3, label="Jev (ours)" if i == 0 else None)
        ax.bar(lx, BEST_LLM[k], w, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="Best frontier LLM (paper)" if i == 0 else None)
        ax.text(jx, JEV[k] + 1.2, f"{JEV[k]:.1f}", ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.text(lx, BEST_LLM[k] + 1.2, f"{BEST_LLM[k]:.1f}", ha="center", va="bottom",
                fontsize=10.5, color=MUTED)
        ax.text(lx, 1.5, LLM_SRC[k], ha="center", va="bottom", fontsize=7, color=MUTED, rotation=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{k}\n(step)" if k == "When" else f"{k}\n(agent)" if k == "Who"
         else f"{k}\n(error F1)" if k == "What" else f"{k}\n(joint)" for k in AXES],
        fontsize=10, color=INK)
    ax.set_ylim(0, 90)
    ax.set_ylabel("accuracy / macro-F1  (%)", fontsize=10, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(length=0, colors=MUTED)
    ax.yaxis.grid(True, color="#ecebE7", linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Finding what broke an AI agent: Jev vs frontier LLMs",
                 fontsize=15, color=INK, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.02, "Who&When Pro, text subset (6,257 traces). Higher is better.",
            transform=ax.transAxes, fontsize=10, color=MUTED)

    ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=1)
    fig.text(0.075, 0.055,
             "Jev ran all 6,257 traces for ~$1.28 (input-only; output tokens free).  "
             "Hatched = constrained-choice for Jev: it picks from the trace's listed agents/steps,",
             fontsize=7.2, color=MUTED)
    fig.text(0.075, 0.018,
             "while the LLMs free-generate. What (error macro-F1) is the like-for-like axis.  "
             "Baselines: best of 4 frontier LLMs per axis, arXiv:2607.09996 Table 4.",
             fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.19)
    out = Path("figures/whowhen_jev_vs_llm.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
