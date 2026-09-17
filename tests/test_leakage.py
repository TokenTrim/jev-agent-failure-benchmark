"""No ground-truth label or gold answer reaches a model input.

Runs on real traces when the dataset is present; the renderers are what could
leak, so the check must exercise them.
"""
from pathlib import Path

from conftest import needs_dataset

FORBIDDEN = ("is_injected", "is_replayed", "accepted_predictions", "ground_truth")


@needs_dataset
def test_prompt_and_state_carry_no_labels():
    from whowhen_eval.prompts import load_taxonomy, parts_to_text
    from whowhen_eval.run import build_prompt

    from jevbench.backends.jev import _state
    from jevbench.candidates import candidate_steps  # noqa: F401 (import guard)
    from jevbench.dataset import load_examples, release_of

    root = Path(__file__).resolve().parent.parent / "data"
    taxonomy = load_taxonomy(root)
    examples = load_examples(root / "text.jsonl")

    seen: set[str] = set()
    checked = 0
    for ex in examples:
        if ex.framework in seen:
            continue
        seen.add(ex.framework)
        release = release_of(ex, root)
        prompt = parts_to_text(build_prompt(release, ex.framework, taxonomy))
        transcript_state = _state((release.get("task") or {}).get("query") or "", "")

        for needle in FORBIDDEN:
            assert needle not in prompt, f"{ex.framework}: prompt leaks {needle!r}"

        answer = (release.get("task") or {}).get("answer")
        if isinstance(answer, str) and len(answer.strip()) >= 4:
            assert answer not in prompt, f"{ex.framework}: prompt leaks task.answer"
            assert answer not in transcript_state
        checked += 1
    assert checked >= 5
