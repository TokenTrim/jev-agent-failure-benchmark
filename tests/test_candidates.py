from conftest import make_render, needs_dataset

from jevbench.candidates import candidate_agents, candidate_steps


def test_extracts_agents_and_steps_in_order():
    rr = make_render(agents=["node_1", "node_2", "node_1"], steps=["1.0", "1.1", "2.0"])
    assert candidate_agents(rr) == ["node_1", "node_2"]  # distinct, first-seen order
    assert candidate_steps(rr) == ["1.0", "1.1", "2.0"]


def test_single_agent_fallback():
    rr = make_render(agents=["agent"], steps=["1"])
    assert candidate_agents(rr) == ["agent"]


def test_colon_header_shape():
    from whowhen_eval.render.base import RenderResult, TranscriptBlock
    rr = RenderResult(
        blocks=[TranscriptBlock(coord="3", text="step 3: solver: reasoning here")],
        step_index=[("3", (3,))],
    )
    assert candidate_agents(rr) == ["solver"]


@needs_dataset
def test_a_candidate_step_scores_correct_on_real_traces():
    """A correct answer must be reachable from the offered candidate steps.

    For each real trace, at least one candidate step should score `step=True`
    under the framework's own official scorer (which recomposes the GT
    coordinate into the rendered space, e.g. mathchat maps `1.1` -> `3`).
    Asserts high overall coverage and prints any misses.
    """
    from collections import Counter
    from pathlib import Path

    from whowhen_eval.render import get_renderer
    from whowhen_eval.score import score

    from jevbench.dataset import load_examples, release_of

    root = Path(__file__).resolve().parent.parent / "data"
    examples = load_examples(root / "text.jsonl")

    per_fw: Counter = Counter()
    hits = 0
    total = 0
    misses: list[str] = []
    for ex in examples:
        if per_fw[ex.framework] >= 5:  # a few per framework, all frameworks
            continue
        per_fw[ex.framework] += 1
        release = release_of(ex, root)
        rr = get_renderer(ex.framework)(release)
        gt = release["ground_truth"]
        reachable = any(
            score({"step_coord": s}, gt, ex.framework)["step"]
            for s in candidate_steps(rr)
        )
        total += 1
        if reachable:
            hits += 1
        else:
            misses.append(f"{ex.framework}:{ex.id}")
    coverage = hits / total
    print(f"candidate-step coverage: {hits}/{total} = {coverage:.2%}; misses={misses[:10]}")
    assert len(per_fw) >= 5
    assert coverage >= 0.9, f"low candidate-step coverage {coverage:.2%}: {misses[:10]}"
