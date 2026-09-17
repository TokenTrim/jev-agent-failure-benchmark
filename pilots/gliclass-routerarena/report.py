"""Score the routing decisions against RouterArena's per-model numbers.

Routed accuracy, cost, and latency are the picked models' cached RouterArena
values, averaged over the successfully-routed examples. Baselines: each fixed
model, always-cheapest, best-fixed (hindsight), and the oracle ceiling. Costs
and latencies are RouterArena's reported figures, not measured GLiClass calls.
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

from labels import LABELS, REAL_ID

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "sample.json"
RESULTS = HERE / "results" / "routes.jsonl"
OUT = HERE / "results" / "summary.json"


def _load_routes() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["global_index"]] = rec
    return latest


def _mean(xs: list[float]) -> float | None:
    return st.mean(xs) if xs else None


def compute() -> dict:
    examples = {e["global_index"]: e for e in json.loads(SAMPLE.read_text())["examples"]}
    routes = _load_routes()
    models = list(LABELS)

    ok = {gi: r for gi, r in routes.items() if not r.get("error") and r.get("pick_alias")}
    failed = [gi for gi, r in routes.items() if r.get("error")]
    scored_ids = [gi for gi in ok if gi in examples]

    # routed metrics over successfully-routed examples
    routed_score, routed_cost, routed_lat = [], [], []
    for gi in scored_ids:
        pick = ok[gi]["pick_alias"]
        m = examples[gi]["models"].get(pick)
        if not m:
            continue
        if m["score"] is not None:
            routed_score.append(m["score"])
        if m["cost_usd"] is not None:
            routed_cost.append(m["cost_usd"])
        if m["latency_ms"] is not None:
            routed_lat.append(m["latency_ms"])

    # fixed-model baselines over the SAME examples
    fixed: dict[str, dict] = {}
    for mid in models:
        s = [examples[gi]["models"][mid]["score"] for gi in scored_ids
             if examples[gi]["models"].get(mid) and examples[gi]["models"][mid]["score"] is not None]
        c = [examples[gi]["models"][mid]["cost_usd"] for gi in scored_ids
             if examples[gi]["models"].get(mid) and examples[gi]["models"][mid]["cost_usd"] is not None]
        latv = [examples[gi]["models"][mid]["latency_ms"] for gi in scored_ids
                if examples[gi]["models"].get(mid) and examples[gi]["models"][mid]["latency_ms"] is not None]
        fixed[mid] = {"acc": _mean(s), "cost_usd": _mean(c), "latency_ms": _mean(latv),
                      "real_id": REAL_ID.get(mid)}

    cheapest = min(fixed, key=lambda m: fixed[m]["cost_usd"])
    best_fixed = max(fixed, key=lambda m: fixed[m]["acc"])
    oracle = _mean([max(examples[gi]["models"][mid]["score"] for mid in examples[gi]["models"]
                        if examples[gi]["models"][mid]["score"] is not None) for gi in scored_ids])

    dist = Counter(ok[gi]["pick_alias"] for gi in scored_ids)
    api_lat = [routes[gi]["latency_s"] for gi in ok if routes[gi].get("latency_s") is not None]

    return {
        "n_sample": len(examples),
        "n_routed_ok": len(scored_ids),
        "n_failed": len(failed),
        "completion_rate": round(len(scored_ids) / len(examples), 4) if examples else 0,
        "failed_ids": failed,
        "routed": {"acc": _mean(routed_score), "cost_usd": _mean(routed_cost),
                   "latency_ms": _mean(routed_lat)},
        "fixed": fixed,
        "always_cheapest": {"model": cheapest, **fixed[cheapest]},
        "best_fixed_hindsight": {"model": best_fixed, **fixed[best_fixed]},
        "oracle_ceiling_acc": oracle,
        "selection_distribution": dict(dist),
        "gliclass_api_latency_s": {"median": _mean(api_lat) and round(st.median(api_lat), 2),
                                   "n": len(api_lat)},
        "notes": "Costs/latencies are RouterArena reported per-model figures, not measured "
                 "GLiClass calls. GLiClass is a zero-shot classifier, not Jev/OpenJev.",
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(r: dict) -> str:
    rt = r["routed"]
    L = ["## Results (GLiClass routing pilot on RouterArena)", "",
         (f"Sample: {r['n_sample']} fixed-seed RouterArena examples (6-model pool). "
          f"Completion {_pct(r['completion_rate'])} ({r['n_routed_ok']} routed, {r['n_failed']} failed)."),
         "", "| Policy | Accuracy | Cost / query | Latency |",
         "|---|---|---|---|",
         f"| **GLiClass routed** | {_pct(rt['acc'])} | ${rt['cost_usd']:.6f} | {rt['latency_ms']:.0f} ms |"]
    for mid, f in sorted(r["fixed"].items(), key=lambda kv: -kv[1]["acc"]):
        tag = []
        if mid == r["always_cheapest"]["model"]:
            tag.append("cheapest")
        if mid == r["best_fixed_hindsight"]["model"]:
            tag.append("best-fixed, hindsight")
        label = f"{mid}" + (f" ({', '.join(tag)})" if tag else "")
        L.append(f"| fixed: {label} | {_pct(f['acc'])} | ${f['cost_usd']:.6f} | {f['latency_ms']:.0f} ms |")
    L.append(f"| oracle ceiling (per-query best) | {_pct(r['oracle_ceiling_acc'])} | - | - |")
    L += ["", f"Selection distribution: {r['selection_distribution']}.", "",
          ("Costs and latencies are RouterArena's reported per-model figures (not measured "
           "GLiClass calls). This is a small GLiClass pilot, not a leaderboard result, and not "
           "Jev or OpenJev. No production cost-savings claim is made from a free demo.")]
    return "\n".join(L) + "\n"


def main() -> int:
    r = compute()
    OUT.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(render_markdown(r))
    print(f"summary.json -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
