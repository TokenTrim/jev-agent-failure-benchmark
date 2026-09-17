"""Jev backend: payload assembly and answer mapping, fully offline (mocked)."""
from conftest import make_render
from whowhen_eval.prompts import Taxonomy

from jevbench.backends import jev

TAX = Taxonomy(
    codes=("V.1", "V.2"),
    block="- V.1: **Context/memory Loss** - lost context\n"
          "- V.2: **Inadequate Or Incorrect Verification** - did not verify",
)

CANNED = {
    "model": "jev-1.13.0",
    "answers": {
        "agent": {"type": "choice", "choice": "node_1",
                  "probabilities": {"node_1": 0.7, "node_2": 0.3}, "confidence": 0.6},
        "step": {"type": "choice", "choice": "1.1",
                 "probabilities": {"1.0": 0.2, "1.1": 0.8}, "confidence": 0.7},
        "mode": {"type": "choice", "choice": "V.2",
                 "probabilities": {"V.1": 0.4, "V.2": 0.6}, "confidence": 0.55},
    },
    "usage": {"input_tokens": 1234, "output_tokens": 40},
}


def test_payload_and_mapping(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    rr = make_render(agents=["node_1", "node_2"], steps=["1.0", "1.1"])
    monkeypatch.setattr(jev, "get_renderer", lambda fw: (lambda release: rr))

    captured = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return CANNED

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return Resp()

    monkeypatch.setattr(jev.httpx, "post", fake_post)

    release = {"task": {"query": "solve it", "answer": "SECRET-ANSWER"},
               "ground_truth": {"agent": "node_1", "step": "1.1", "mode": "V.2"}}
    backend = jev.JevBackend()
    pred = backend.predict(release, "magentic-one", TAX)

    payload = captured["json"]
    state = payload["state"]
    assert "## Transcript" in state and "Agent: node_1" in state
    assert "ground_truth" not in state and "SECRET-ANSWER" not in state  # no label leak
    q = payload["questions"]
    assert list(q["agent"]["criteria"]) == ["node_1", "node_2"]
    assert list(q["step"]["criteria"]) == ["1.0", "1.1"]
    assert set(q["mode"]["criteria"]) == {"V.1", "V.2"}
    assert captured["headers"]["Authorization"] == "Bearer test-key"

    assert pred.error is None
    assert pred.pred == {"agent_name": "node_1", "step_coord": "1.1",
                         "error_mode": "V.2", "reason": None, "parse_warnings": []}
    assert pred.probabilities["mode"] == {"V.1": 0.4, "V.2": 0.6}
    assert pred.confidence["step"] == 0.7
    assert pred.usage["input_tokens"] == 1234


def test_missing_key_is_recorded_not_raised(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    rr = make_render(agents=["a"], steps=["1"])
    monkeypatch.setattr(jev, "get_renderer", lambda fw: (lambda release: rr))
    pred = jev.JevBackend().predict({"task": {}, "ground_truth": {}}, "smolagents", TAX)
    assert pred.pred is None and "TYPESAFE_API_KEY" in pred.error
