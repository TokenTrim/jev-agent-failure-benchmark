"""Route the sampled requests through GLiClass, one at a time, resumable.

Only the request text and the frozen capability labels go to the classifier.
Cached scores, costs, and ground truth stay out of the classifier input. Each
routing decision is appended immediately so the run resumes after an
interruption. Stops early if the service returns too many consecutive errors.
"""
from __future__ import annotations

import json
from pathlib import Path

from gliclass_router import make_client, route
from labels import LABEL_TO_ALIAS, LABELS, ROUTING_PROMPT

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "sample.json"
RESULTS = HERE / "results" / "routes.jsonl"
MAX_CONSECUTIVE_ERRORS = 5


def _done_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    out = set()
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("error"):
                out.add(rec["global_index"])
    return out


def main() -> int:
    examples = json.loads(SAMPLE.read_text())["examples"]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids()
    todo = [e for e in examples if e["global_index"] not in done]
    print(f"routing {len(todo)} / {len(examples)} (resuming, {len(done)} done)")

    label_list = list(LABELS.values())
    client = make_client()
    consecutive_errors = 0
    for i, ex in enumerate(todo):
        r = route(client, ex["request"], label_list, ROUTING_PROMPT)
        pick_alias = LABEL_TO_ALIAS.get(r.pick_label) if r.pick_label else None
        rec = {
            "global_index": ex["global_index"],
            "pick_label": r.pick_label,
            "pick_alias": pick_alias,
            "scores": {LABEL_TO_ALIAS.get(k, k): v for k, v in r.scores.items()},
            "latency_s": r.latency_s,
            "error": r.error,
        }
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if r.error:
            consecutive_errors += 1
            print(f"  [{i + 1}/{len(todo)}] {ex['global_index']} ERROR {r.error}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("stopping: too many consecutive errors (service may be unavailable)")
                return 1
        else:
            consecutive_errors = 0
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(todo)}] {ex['global_index']} -> {pick_alias} ({r.latency_s}s)")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
