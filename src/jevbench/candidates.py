"""Candidate agent ids and step coordinates, read from the rendered transcript.

Jev answers with typed `choice` questions, so it needs the option sets. We take
them from the same rendered transcript the LLM baseline sees (never from the raw
trajectory fields or ground truth), so both models work from identical evidence.
Step options are the exact coordinate strings the prompt uses; agent options are
the ids that appear in the step headers.
"""
from __future__ import annotations

import re

from whowhen_eval.render.base import TASK_ANCHOR, RenderResult

# Two header shapes across renderers: "Step 1.2 | Agent: node_1" (gui, smolagents,
# alfagent, magentic) and "step 3: agent_name: ..." (base flat/hier helpers).
_AGENT_PIPE = re.compile(r"Agent:\s*(\S+)")
_AGENT_COLON = re.compile(r"^\s*step\s+\S+\s*:\s*([^:]+?)\s*:", re.IGNORECASE)


def _agent_of_block_text(text: str) -> str | None:
    """Pull the agent id from a step block's first line, or None."""
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    m = _AGENT_PIPE.search(first)
    if m:
        return m.group(1).strip()
    m = _AGENT_COLON.match(first)
    if m:
        return m.group(1).strip()
    return None


def candidate_steps(rr: RenderResult) -> list[str]:
    """The valid step coordinate strings, in transcript order."""
    return [coord for coord, _ in rr.step_index]


def candidate_agents(rr: RenderResult) -> list[str]:
    """Distinct agent ids appearing in step headers, in first-seen order.

    Falls back to `["agent"]` when no header exposes an id, which matches the
    label single-agent renderers hardcode. The scorer excludes single-agent
    frameworks from the Who metric anyway.
    """
    seen: list[str] = []
    for block in rr.blocks:
        if block.coord == TASK_ANCHOR or not block.text:
            continue
        agent = _agent_of_block_text(block.text)
        if agent and agent not in seen:
            seen.append(agent)
    return seen or ["agent"]
