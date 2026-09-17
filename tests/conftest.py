"""Shared fixtures: synthetic render results and result records (no network)."""
from __future__ import annotations

from pathlib import Path

import pytest
from whowhen_eval.render.base import RenderResult, TranscriptBlock

DATA = Path(__file__).resolve().parent.parent / "data" / "text.jsonl"
needs_dataset = pytest.mark.skipif(not DATA.exists(), reason="data/text.jsonl not downloaded")


def make_render(agents: list[str], steps: list[str]) -> RenderResult:
    """A minimal multi-agent RenderResult with `Step C | Agent: X` headers."""
    blocks = [
        TranscriptBlock(coord=s, text=f"Step {s} | Agent: {a}\n[output] did a thing")
        for s, a in zip(steps, agents)
    ]
    return RenderResult(
        blocks=blocks,
        step_format_hint="step R.P",
        step_index=[(s, tuple(int(x) for x in s.split("."))) for s in steps],
        trajectory_length=len(steps),
    )


def record(fw: str, *, agent: bool, step: bool, mode: bool, gt_mode: str,
           pr_mode: str, error: str | None = None, latency: float = 1.0,
           in_tok: int = 100, out_tok: int = 50,
           probs: dict | None = None) -> dict:
    """One result record in the schema the runner writes."""
    return {
        "trace_id": f"{fw}-{gt_mode}-{step}-{agent}-{id(object())}",
        "framework": fw,
        "benchmark": "b",
        "modality": "text",
        "backend": "test",
        "model": "test",
        "prediction": {"agent_name": "a" if agent else "x", "step_coord": "1",
                       "error_mode": pr_mode, "reason": None, "parse_warnings": []},
        "score": {"agent": agent, "step": step, "mode": mode},
        "ground_truth": {"agent": "a", "step": "1", "mode": gt_mode},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                  "total_tokens": in_tok + out_tok},
        "latency_s": latency,
        "probabilities": probs or {},
        "confidence": {},
        "error": error,
    }
