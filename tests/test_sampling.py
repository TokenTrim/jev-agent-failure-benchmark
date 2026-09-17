from jevbench.dataset import Example, stratified_sample


def _examples() -> list[Example]:
    out = []
    for fw, n in (("smolagents", 100), ("magentic-one", 30), ("metagpt", 5)):
        out += [Example(f"{fw}_{i}", fw, "b", {"id": f"{fw}_{i}"}) for i in range(n)]
    return out


def test_deterministic_under_seed():
    ex = _examples()
    a = stratified_sample(ex, 60, seed=7, floor=5)
    b = stratified_sample(ex, 60, seed=7, floor=5)
    assert a == b
    assert stratified_sample(ex, 60, seed=8, floor=5) != a


def test_floor_covers_every_framework():
    ex = _examples()
    ids = set(stratified_sample(ex, 60, seed=1, floor=5))
    by_fw = {}
    for e in ex:
        if e.id in ids:
            by_fw[e.framework] = by_fw.get(e.framework, 0) + 1
    assert by_fw["metagpt"] == 5  # floor honored even though it is the smallest
    assert by_fw["smolagents"] >= 5


def test_meets_requested_n_exactly():
    ex = _examples()  # 135 rows across 3 frameworks
    assert len(stratified_sample(ex, 60, seed=1, floor=5)) == 60
    # n at/over the dataset size returns every id, not a rounded-down subset.
    assert set(stratified_sample(ex, len(ex), seed=1, floor=5)) == {e.id for e in ex}
    assert set(stratified_sample(ex, 10_000, seed=1, floor=5)) == {e.id for e in ex}


def test_floor_capped_by_framework_size():
    ex = [Example(f"x_{i}", "x", "b", {}) for i in range(3)]
    ids = stratified_sample(ex, 10, seed=1, floor=20)
    assert len(ids) == 3  # cannot exceed available rows
