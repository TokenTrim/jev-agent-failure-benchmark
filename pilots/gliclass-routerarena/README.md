# GLiClass routing pilot on RouterArena

A small pilot: can a **zero-shot text classifier** (GLiClass, via the hosted
`knowledgator/GLiClass_SandBox` Space) act as an **LLM router** — read a request
and pick which model should answer it — and does that beat fixed single-model
baselines on [RouterArena](https://arxiv.org/abs/2510.00202)?

This is a **GLiClass pilot, not a leaderboard result, and not Jev or OpenJev.**
GLiClass is knowledgator's zero-shot classifier; it is unrelated to Typesafe's
Jev. No production cost-savings claim is made from a free demo.

## Result: the router did not beat the best fixed model

GLiClass collapsed to essentially one model, picking `80b` for **98 of 100**
requests, and landed at **61.5%** accuracy — below the best single model `235b`
(**66.4%**, also the cheapest on this sample) and below `gemini-flash`
(**64.3%**). It beat only the three weakest models. The **oracle ceiling
(81.3%)** shows large routing headroom this zero-shot, capability-label router
did not capture. On this sample, **picking the single best model beats this
router.**

| Policy | Accuracy | Cost / query | Latency |
|---|---|---|---|
| **GLiClass routed** | 61.5% | $0.000169 | 2510 ms |
| fixed: 235b (cheapest, best-fixed, hindsight) | 66.4% | $0.000030 | 18046 ms |
| fixed: gemini-flash | 64.3% | $0.002999 | 5814 ms |
| fixed: 80b | 60.5% | $0.000131 | 2480 ms |
| fixed: 30b | 60.3% | $0.000223 | 13487 ms |
| fixed: coder-next | 60.0% | $0.000301 | 6499 ms |
| fixed: haiku | 47.9% | $0.000172 | 1705 ms |
| oracle ceiling (per-query best) | 81.3% | - | - |

Selection distribution: `{80b: 98, gemini-flash: 2}`. Completion 100/100, 0
failed. Full numbers in `results/summary.json`, per-request decisions in
`results/routes.jsonl`. Costs/latencies are RouterArena's reported per-model
figures, not measured GLiClass calls.

## Method

- **Sample:** 100 fixed-seed RouterArena examples (seed `20240517`), drawn from
  the 6-model derived set (see Data). Sampled ids are saved in
  `data/sample.json`.
- **Router input:** only the request text plus six **frozen capability labels**
  (`labels.py`), fixed before evaluation. The classifier never sees cached model
  answers, per-model scores, or ground truth.
- **Routing:** one GLiClass call per request (`multi_label=True`, argmax over the
  label scores) maps to a model id.
- **Scoring:** the picked model's **cached RouterArena score** is the routed
  answer's correctness. Accuracy, cost, and latency come from RouterArena's
  per-model figures — the GLiClass demo's own call time is not the routing cost.
- **Baselines:** each fixed model, **always-cheapest**, **best-fixed (labeled
  hindsight)**, and the **oracle ceiling** (per-query best model).
- No tuning on RouterArena examples or results; labels and prompt frozen up front.

## Reproduce

```sh
uv run --project ../.. python build_sample.py   # fetch + cache the fixed-seed sample
uv run --project ../.. python run_pilot.py      # route via GLiClass (sequential, resumable)
uv run --project ../.. python report.py         # score + write results/summary.json
```

`run_pilot.py` sends one request at a time, retries with backoff, saves each
decision to `results/routes.jsonl`, resumes on rerun, and stops if the Space
returns repeated errors.

## Data and attribution

- **RouterArena** — Lu et al., *"RouterArena: An Open Platform for Comprehensive
  Comparison of LLM Routers"*, arXiv:2510.00202 (2025);
  [github.com/RouteWorks/RouterArena](https://github.com/RouteWorks/RouterArena)
  (Apache-2.0).
- Per-model scores/costs/latency for the 6-model pool come from the derived set
  [`RaghavendraSqwish/routerarena-with-embeddings`](https://huggingface.co/datasets/RaghavendraSqwish/routerarena-with-embeddings)
  (revision `96a3adb`), which is research-use. We do **not** redistribute its
  rows here; `build_sample.py` fetches them and we commit only our sampled ids
  and routing decisions.
- **GLiClass** — [knowledgator/GLiClass_SandBox](https://huggingface.co/spaces/knowledgator/GLiClass_SandBox).

## Caveats

- RouterArena's reported per-model **costs look noisy** (one open model is priced
  far below the rest); treat cost figures as indicative, not audited.
- Latency is RouterArena's reported per-model latency, not a live measurement.
- Small pool (6 models) and small sample (100). Indicative only.
