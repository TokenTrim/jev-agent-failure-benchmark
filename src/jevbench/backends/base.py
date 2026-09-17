"""The backend interface and the prediction it returns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from whowhen_eval.prompts import Taxonomy


@dataclass
class Prediction:
    """One backend's answer for one trace.

    `pred` holds the three scored fields (`agent_name`, `step_coord`,
    `error_mode`) plus `reason`, ready for `whowhen_eval.score`. `probabilities`
    and `confidence` are per-axis and populated only when the backend exposes
    them (Jev does; a temperature-0 LLM does not), so calibration is measurable
    for whichever backend supplies them.
    """

    pred: dict[str, Any] | None
    usage: dict[str, int | None]
    latency_s: float
    raw: Any = None
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class Backend(Protocol):
    """A model that predicts (agent, step, mode) for a rendered trace."""

    name: str
    model: str

    def predict(
        self, release: dict[str, Any], framework: str, taxonomy: Taxonomy
    ) -> Prediction:
        """Return a `Prediction` for one release; must not raise on model error.

        Network or provider failures are caught and returned as
        `Prediction(pred=None, ..., error=...)` so the runner records the
        failure instead of dropping the trace.
        """
        ...
