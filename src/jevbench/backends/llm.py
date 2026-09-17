"""The LLM baseline, run under the exact official all-at-once protocol.

This backend changes nothing about the official protocol: it builds the same
prompt (`build_prompt`), calls the model through the same LiteLLM wrapper
(`whowhen_eval.llm.generate`), and parses with the same lenient regex parser
(`parse_all_at_once`). Only the timing and the `Prediction` envelope are ours.
"""
from __future__ import annotations

import time
from typing import Any

from whowhen_eval.llm import generate
from whowhen_eval.parse import parse_all_at_once
from whowhen_eval.prompts import Taxonomy, user_msg
from whowhen_eval.run import build_prompt

from .base import Prediction


class LLMBackend:
    """Baseline LLM via LiteLLM. `model` is any id LiteLLM routes (e.g. gpt-5.6)."""

    name = "llm"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def predict(
        self, release: dict[str, Any], framework: str, taxonomy: Taxonomy
    ) -> Prediction:
        parts = build_prompt(release, framework, taxonomy)
        t0 = time.monotonic()
        try:
            raw, usage = generate(
                self.model,
                [user_msg(parts)],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
        except Exception as e:  # noqa: BLE001 — recorded, not raised
            return Prediction(
                pred=None,
                usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
                latency_s=round(time.monotonic() - t0, 3),
                error=f"{type(e).__name__}: {e}",
            )
        latency = round(time.monotonic() - t0, 3)

        parsed = parse_all_at_once(raw, taxonomy.codes)
        pred = {
            "agent_name": parsed.agent_name,
            "step_coord": parsed.step_coord,
            "error_mode": parsed.error_mode,
            "reason": parsed.reason,
            "parse_warnings": parsed.parse_warnings,
        }
        return Prediction(pred=pred, usage=usage, latency_s=latency, raw=raw)
