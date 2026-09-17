"""Frozen routing labels for the 6-model RouterArena pool.

Descriptions are capability-only and were fixed BEFORE any evaluation. They
carry no model answers, no scores, and no ground truth. Aliases match the
`model_id` field in the derived dataset's `model_summaries`; `REAL_ID` records
the underlying checkpoint for the report (best-effort mapping, not sent to the
classifier).
"""
from __future__ import annotations

# alias -> descriptive label shown to GLiClass (no model name, no scores).
# No commas: the Space splits the labels field on commas.
LABELS: dict[str, str] = {
    "haiku": "a small fast lightweight assistant best for simple short factual questions",
    "30b": "a small general model good for easy everyday questions",
    "80b": "a mid sized general model good for moderately complex questions",
    "235b": "a very large high capability model best for hard reasoning and math and multi step problems",
    "coder-next": "a programming specialized model best for coding and code generation tasks",
    "gemini-flash": "a fast broad general purpose model good for a wide range of everyday tasks",
}

# best-effort underlying checkpoint, for the report only (not classifier input)
REAL_ID: dict[str, str] = {
    "haiku": "claude-3-haiku-20240307",
    "30b": "qwen3-30b-a3b-instruct-2507",
    "80b": "qwen3-next-80b-a3b-instruct",
    "235b": "qwen3-235b-a22b-2507",
    "coder-next": "qwen3-coder-next",
    "gemini-flash": "gemini-2.5-flash",
}

# Frozen routing instruction given to the classifier.
ROUTING_PROMPT = "Select the single model best suited to correctly answer this request:"

# label string -> alias (labels are unique)
LABEL_TO_ALIAS: dict[str, str] = {v: k for k, v in LABELS.items()}
