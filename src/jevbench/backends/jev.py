"""The Jev backend: the same trace as three typed `choice` questions.

Jev is a decision model, not a text generator. It cannot free-write a step
coordinate, so we hand it the option sets the LLM would otherwise produce from
the transcript: the agent ids and step coordinates that appear in the rendered
trace, and the taxonomy codes that appear in the prompt for both models. All
three questions go in one call and each returns a calibrated probability
distribution, which is what makes calibration measurable for Jev.

This asymmetry (Jev picks from enumerated candidates; the LLM free-generates) is
the one adaptation the API forces. It is reported separately, and the candidate
sets add no evidence beyond what the shared transcript and taxonomy already
carry.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx
from whowhen_eval.prompts import Taxonomy
from whowhen_eval.render import get_renderer

from ..candidates import candidate_agents, candidate_steps
from .base import Prediction

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_BLOCK_LINE = re.compile(r"^- (?P<code>\S+): \*\*(?P<name>.+?)\*\* - (?P<desc>.*)$")

_AGENT_Q = "Which agent's turn first introduced the decisive error in this failed run?"
_STEP_Q = "At which step coordinate did the first decisive error occur?"
_MODE_Q = "Which taxonomy error mode best matches the first decisive error?"


def _mode_criteria(taxonomy: Taxonomy) -> dict[str, str | None]:
    """Map each taxonomy code to its `name: description` rubric from the block."""
    out: dict[str, str | None] = {}
    for line in taxonomy.block.splitlines():
        m = _BLOCK_LINE.match(line.strip())
        if m:
            out[m.group("code")] = f"{m.group('name')}: {m.group('desc')}"
    for code in taxonomy.codes:  # ensure every code is an option
        out.setdefault(code, None)
    return out


def _state(problem: str, transcript: str) -> str:
    """The shared case text: user question plus the rendered transcript."""
    return f"## User Question\n\n{problem}\n\n## Transcript\n\n{transcript}"


class JevBackend:
    """Typesafe.ai Jev via `POST /v1/systemone`."""

    name = "jev"

    def __init__(
        self,
        model: str = "jev-1.13.0",
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 600.0,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on 429/529, honoring Retry-After."""
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        backoff = 1.0
        last: httpx.HTTPStatusError | None = None
        for _ in range(self.max_retries + 1):
            resp = httpx.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code in (429, 529):
                wait = float(resp.headers.get("retry-after", backoff))
                time.sleep(wait)
                backoff = min(backoff * 2, 30.0)
                last = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
                continue
            resp.raise_for_status()
            return resp.json()
        assert last is not None
        raise last

    def predict(
        self, release: dict[str, Any], framework: str, taxonomy: Taxonomy
    ) -> Prediction:
        t0 = time.monotonic()
        try:
            rr = get_renderer(framework)(release)
            agents = candidate_agents(rr)
            steps = candidate_steps(rr)
            problem = (release.get("task") or {}).get("query") or ""
            questions: dict[str, Any] = {
                "agent": {"type": "choice", "instructions": _AGENT_Q,
                          "criteria": {a: None for a in agents}},
                "mode": {"type": "choice", "instructions": _MODE_Q,
                         "criteria": _mode_criteria(taxonomy)},
            }
            if steps:  # a trace with no step blocks leaves step unanswered
                questions["step"] = {"type": "choice", "instructions": _STEP_Q,
                                     "criteria": {s: None for s in steps}}
            payload = {
                "state": _state(problem, rr.chat_content),
                "model": self.model,
                "questions": questions,
            }
            data = self._post(payload)
        except Exception as e:  # noqa: BLE001 — recorded, not raised
            return Prediction(
                pred=None,
                usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
                latency_s=round(time.monotonic() - t0, 3),
                error=f"{type(e).__name__}: {e}",
            )
        latency = round(time.monotonic() - t0, 3)

        answers = data.get("answers") or {}
        probs: dict[str, dict[str, float]] = {}
        conf: dict[str, float] = {}
        for axis in ("agent", "step", "mode"):
            ans = answers.get(axis)
            if isinstance(ans, dict):
                if isinstance(ans.get("probabilities"), dict):
                    probs[axis] = ans["probabilities"]
                if ans.get("confidence") is not None:
                    conf[axis] = ans["confidence"]

        def choice(axis: str) -> str | None:
            ans = answers.get(axis)
            return ans.get("choice") if isinstance(ans, dict) else None

        pred = {
            "agent_name": choice("agent"),
            "step_coord": choice("step"),
            "error_mode": choice("mode"),
            "reason": None,
            "parse_warnings": [],
        }
        usage_raw = data.get("usage") or {}
        in_tok = usage_raw.get("input_tokens")
        out_tok = usage_raw.get("output_tokens")
        usage = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": (in_tok or 0) + (out_tok or 0) if in_tok is not None else None,
        }
        return Prediction(
            pred=pred, usage=usage, latency_s=latency, raw=data,
            probabilities=probs, confidence=conf,
        )
