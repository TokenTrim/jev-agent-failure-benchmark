"""Load the Who&When Pro text subset and draw a reproducible sample.

The text subset is one JSONL file, `data/text.jsonl`, at a pinned dataset
revision (see README for the download command). Each row carries JSON-encoded
`task`, `trajectory`, `ground_truth`, `extras`. We never read `ground_truth`
or `task.answer` into anything a model sees; those live only in scoring.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whowhen_eval.run import release_from_row

# The dataset revision this project pins. Recorded in every sample manifest.
DATASET_REVISION = "0bd196c8a040841c4ae167ab33cc8151de246f1f"
SPLIT = "text"


@dataclass(frozen=True)
class Example:
    """One trace row, with the label-bearing fields left encoded in `row`."""

    id: str
    framework: str
    benchmark: str
    row: dict[str, Any]


def load_examples(text_jsonl: Path) -> list[Example]:
    """Read every row of the text subset into memory (~72 MB, 6257 rows)."""
    out: list[Example] = []
    with text_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Example(row["id"], row["framework"], row["benchmark"], row))
    return out


def release_of(example: Example, data_root: Path) -> dict[str, Any]:
    """Reconstruct the release dict the official renderers expect."""
    return release_from_row(example.row, SPLIT, data_root)


def stratified_sample(
    examples: list[Example], n: int, seed: int, floor: int = 20
) -> list[str]:
    """Return `n` trace ids, stratified by framework with a per-framework floor.

    Proportional allocation across frameworks, but every framework present gets
    at least `floor` ids (capped at its size) so small frameworks and the Who
    metric keep enough support. Deterministic under `seed`. Returns ids sorted
    for a stable manifest; sampling order does not matter downstream.

    Arguments:
        n: target total sample size.
        seed: RNG seed, saved in the manifest.
        floor: minimum ids per framework.
    """
    by_fw: dict[str, list[str]] = {}
    for ex in examples:
        by_fw.setdefault(ex.framework, []).append(ex.id)

    rng = random.Random(seed)
    picked: set[str] = set()
    # Floor pass: guarantee coverage of every framework first.
    for fw, ids in sorted(by_fw.items()):
        take = min(floor, len(ids))
        picked.update(rng.sample(ids, take))

    # Proportional pass: fill the remaining budget by framework share.
    total = len(examples)
    remaining = max(0, n - len(picked))
    if remaining:
        quota = {
            fw: round(remaining * len(ids) / total) for fw, ids in by_fw.items()
        }
        for fw, ids in sorted(by_fw.items()):
            pool = [i for i in ids if i not in picked]
            take = min(quota.get(fw, 0), len(pool))
            picked.update(rng.sample(pool, take))

    return sorted(picked)


def save_sample(path: Path, ids: list[str], *, seed: int, n: int) -> None:
    """Write the sample manifest: ids plus the seed and dataset revision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "Leoxx/whowhen_pro",
        "revision": DATASET_REVISION,
        "split": SPLIT,
        "seed": seed,
        "requested_n": n,
        "size": len(ids),
        "ids": ids,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_sample(path: Path) -> list[str]:
    """Read the ids from a sample manifest."""
    return json.loads(path.read_text(encoding="utf-8"))["ids"]
