"""Drive one backend over a sample: concurrent, resumable, failures recorded.

Each completed trace is appended to the backend's JSONL immediately, so a run
can be killed and resumed. On resume, trace ids already present with a
non-error record are skipped; errored traces are retried and the later record
wins at read time. The record schema matches `whowhen_eval.leaderboard.collect`
so the official metric recipe reads our files directly.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from whowhen_eval.prompts import Taxonomy
from whowhen_eval.score import score as score_prediction

from .backends.base import Backend
from .dataset import Example, release_of


def done_trace_ids(path: Path) -> set[str]:
    """Trace ids already recorded without an error (safe to skip on resume)."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("trace_id") and not rec.get("error"):
            done.add(rec["trace_id"])
    return done


def _record(backend: Backend, ex: Example, release: dict[str, Any], pred, score) -> dict[str, Any]:
    return {
        "trace_id": ex.id,
        "framework": ex.framework,
        "benchmark": ex.benchmark,
        "modality": "text",
        "backend": backend.name,
        "model": backend.model,
        "prediction": pred.pred,
        "score": score,
        "ground_truth": release.get("ground_truth"),
        "usage": pred.usage,
        "latency_s": pred.latency_s,
        "probabilities": pred.probabilities,
        "confidence": pred.confidence,
        "error": pred.error,
    }


async def run_backend(
    backend: Backend,
    examples: list[Example],
    taxonomy: Taxonomy,
    data_root: Path,
    out_path: Path,
    *,
    concurrency: int = 8,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Run `backend` over `examples`, appending records to `out_path`.

    Returns counts of completed, skipped, and errored traces.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_trace_ids(out_path)
    todo = [ex for ex in examples if ex.id not in done]

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counts = {"completed": 0, "skipped": len(examples) - len(todo), "errored": 0}

    async def worker(ex: Example) -> None:
        async with sem:
            release = release_of(ex, data_root)
            pred = await asyncio.to_thread(backend.predict, release, ex.framework, taxonomy)
        score = None
        if pred.pred is not None:
            score = score_prediction(pred.pred, release.get("ground_truth"), ex.framework)
        rec = _record(backend, ex, release, pred, score)
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["errored" if pred.error else "completed"] += 1
            if on_record:
                on_record(rec)

    await asyncio.gather(*(worker(ex) for ex in todo))
    return counts
