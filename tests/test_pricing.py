from jevbench.pricing import cost_usd, price_for


def test_jev_output_is_free():
    # $0.042 / Mtok input, output billed at 0.
    assert abs(cost_usd("jev-1.13.0", 1_000_000, 5_000_000) - 0.042) < 1e-9


def test_llm_input_and_output():
    # gpt-5.6-terra: $2 in, $12 out per Mtok.
    assert abs(cost_usd("gpt-5.6-terra", 1_000_000, 1_000_000) - 14.0) < 1e-9


def test_unknown_model_is_none():
    assert cost_usd("mystery-model", 1000, 1000) is None
    assert price_for("mystery-model") is None
