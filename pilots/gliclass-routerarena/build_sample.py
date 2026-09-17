"""Fetch a fixed-seed 100-example RouterArena sample and cache it locally.

Source: derived dataset `RaghavendraSqwish/routerarena-with-embeddings`
(6-model pool with per-model score/cost/latency), read via the HF
datasets-server rows API. We keep only what the pilot needs and drop the large
embedding column. Ground-truth answers are not present in this dataset and are
never fetched. The sample manifest records the seed and the exact global ids.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import httpx

DATASET = "RaghavendraSqwish/routerarena-with-embeddings"
CONFIG = "queries_with_embeddings"
SPLIT = "train"
REVISION = "96a3adb46ed27640c36be9f984b8a954a4d3377b"  # pinned; ids recorded regardless
N_ROWS = 8345
ROWS_API = "https://datasets-server.huggingface.co/rows"

HERE = Path(__file__).resolve().parent
SAMPLE_PATH = HERE / "data" / "sample.json"


PAGE = 100


def _fetch_page(client: httpx.Client, offset: int) -> list[dict]:
    """Fetch up to PAGE rows at `offset`, backing off on 429."""
    backoff = 3.0
    for attempt in range(6):
        r = client.get(
            ROWS_API,
            params={"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                    "offset": offset, "length": PAGE},
            timeout=120,
        )
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", backoff))
            time.sleep(wait)
            backoff = min(backoff * 2, 60.0)
            continue
        r.raise_for_status()
        return [x["row"] for x in r.json()["rows"]]
    raise RuntimeError(f"gave up fetching offset {offset} after repeated 429")


def _slim(row: dict) -> dict:
    """Keep request text + per-model score/cost/latency; drop embedding."""
    models = {}
    for s in row.get("model_summaries") or []:
        models[s["model_id"]] = {
            "score": s.get("official_metric_score_mean"),
            "cost_usd": s.get("cost_usd_mean"),
            "latency_ms": s.get("latency_ms_mean"),
            "metric": s.get("official_metric_name"),
        }
    return {
        "global_index": row["global_index"],
        "request": row["prompt_formatted"],
        "track": row.get("track"),
        "models": models,
    }


def build(n: int = 100, seed: int = 20240517, pages: int = 15) -> None:
    """Fetch `pages` spread-out pages, then seeded-sample `n` examples from them."""
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_pages_total = (N_ROWS + PAGE - 1) // PAGE
    page_offsets = sorted(p * PAGE for p in rng.sample(range(n_pages_total), pages))
    pool: list[dict] = []
    with httpx.Client() as client:
        for i, off in enumerate(page_offsets):
            pool.extend(_slim(r) for r in _fetch_page(client, off))
            print(f"  page {i + 1}/{pages} (offset {off}), pool={len(pool)}")
            time.sleep(1.5)
    # seeded sample of n distinct examples from the fetched pool
    by_id = {r["global_index"]: r for r in pool}
    ids = sorted(by_id)
    picked_ids = sorted(rng.sample(ids, min(n, len(ids))))
    rows = [by_id[i] for i in picked_ids]
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "seed": seed,
        "n": n,
        "global_indexes": [r["global_index"] for r in rows],
        "examples": rows,
    }
    SAMPLE_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} examples -> {SAMPLE_PATH}")


if __name__ == "__main__":
    build()
