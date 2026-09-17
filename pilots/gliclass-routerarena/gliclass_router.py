"""Route one request through the hosted GLiClass Space (zero-shot classifier).

Verified API (2026-09-17), Space `knowledgator/GLiClass_SandBox`,
`api_name="/classify_wrapper"`: inputs (text, labels, threshold, multi_label,
prompt, examples, hierarchical, fmt); returns (label_dict, hierarchical_str,
json_str). We use `fmt="json"` and read per-label `scores` from the JSON, then
argmax. The model is not hosted or downloaded here; every call hits the Space.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from gradio_client import Client

SPACE = "knowledgator/GLiClass_SandBox"
API_NAME = "/classify_wrapper"


@dataclass
class Routing:
    pick_label: str | None
    scores: dict[str, float]
    latency_s: float
    error: str | None = None


def make_client() -> Client:
    return Client(SPACE, verbose=False)


def route(
    client: Client,
    text: str,
    labels: list[str],
    prompt: str,
    *,
    max_retries: int = 4,
) -> Routing:
    """Classify `text` over `labels`; return the argmax label and all scores.

    One request at a time. Bounded retries with exponential backoff. The label
    string sent to the Space is the comma-joined `labels`; scores come back keyed
    by those same strings. Network + queue time is included in `latency_s`.
    """
    labels_arg = ", ".join(labels)
    backoff = 2.0
    last_err = ""
    t0 = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            res = client.predict(
                text=text,
                labels=labels_arg,
                threshold=0.0,
                multi_label=True,  # return a score for every label, then argmax
                prompt=prompt,
                examples="",
                hierarchical=False,
                fmt="json",
                api_name=API_NAME,
            )
            scores = json.loads(res[2] or "{}").get("scores", {})
            latency = round(time.monotonic() - t0, 3)
            if not scores:
                return Routing(None, {}, latency, error="empty scores")
            pick = max(scores, key=scores.get)
            return Routing(pick, scores, latency)
        except Exception as e:  # noqa: BLE001 — recorded, retried
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
    return Routing(None, {}, round(time.monotonic() - t0, 3), error=last_err)
