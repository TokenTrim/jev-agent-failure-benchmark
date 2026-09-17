# jev-agent-failure-benchmark

Can a fast typed-decision model find the step that broke an AI agent as accurately
as a strong LLM, at lower latency and cost? This project benchmarks
[**Jev**](https://typesafe.ai) (Typesafe.ai's decision model) against a strong
baseline LLM on the **text subset of Who&When Pro**, a failure-attribution
benchmark for agent systems.

The three predictions per trace are: the **responsible agent**, the **decisive
step**, and the **error category**, scored with the official Who&When Pro recipe.

> **What Who&When Pro measures.** Its failures are *injected* by a controlled
> pipeline that replays a successful trajectory and inserts a single error at a
> chosen point. These are not natural production incidents. Results here are
> about attributing injected faults, and are not a leaderboard submission unless
> the protocol matches exactly (see [Protocol fidelity](#protocol-fidelity)).

## How it works

Both models see the **same trace, rendered the same way**, using the official
`whowhen_eval` harness for rendering, parsing, and scoring:

- **Baseline LLM** runs the exact official *all-at-once* protocol: one prompt,
  free-text answer, parsed by the official lenient parser.
- **Jev** answers three typed `choice` questions in one call (agent, step,
  error mode). It cannot free-write a coordinate, so it picks from the agent ids
  and step coordinates that appear in the same rendered transcript, and the
  taxonomy codes that appear in the prompt for both models. Each choice returns a
  calibrated probability distribution, so Jev's calibration is measured too.

That asymmetry (Jev picks from enumerated candidates; the LLM free-generates) is
the one adaptation the API forces. It is reported separately; the candidate sets
add no information beyond the shared transcript and taxonomy.

Ground-truth labels (`ground_truth.agent/step/mode`) and `task.answer` never
enter a model input. A test enforces this (`tests/test_leakage.py`).

## Setup

```sh
uv venv && uv pip install -e ".[dev,viz]"
cp .env.template .env          # then fill in TYPESAFE_API_KEY and OPENAI_API_KEY
```

Keys are read from the environment (`TYPESAFE_API_KEY`, and the baseline's key,
e.g. `OPENAI_API_KEY`). `.env` is gitignored.

### Get the dataset

The dataset is **not redistributed here**; download the pinned text subset
(~72 MB) directly from Hugging Face:

```sh
mkdir -p data
REV=0bd196c8a040841c4ae167ab33cc8151de246f1f
curl -L "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/$REV/data/text.jsonl" -o data/text.jsonl
curl -L "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/$REV/taxonomy.yaml" -o data/taxonomy.yaml
```

## Run

```sh
# 1. Draw a fixed-seed, framework-stratified sample and save the ids
jevbench sample --n 300 --seed 20240517 --out results/run/sample.json

# 2. Offline cost estimate for that sample (no API calls)
jevbench estimate --sample results/run/sample.json --llm-model gpt-5.6-terra

# 3. Smoke test: a few traces end to end
jevbench sample --n 10 --seed 1 --out results/smoke/sample.json
jevbench --run-dir results/smoke run --sample results/smoke/sample.json \
  --backend jev --backend llm --llm-model gpt-5.6-terra --concurrency 4

# 4. Full run (resumable; rerun to continue after an interruption)
jevbench run --sample results/run/sample.json \
  --backend jev --backend llm --llm-model gpt-5.6-terra --concurrency 8

# 5. Report + chart
jevbench report --llm-model gpt-5.6-terra --chart results/run/chart.png
```

Predictions, token usage, latency, and per-axis probabilities are written
incrementally to `results/<run>/<backend>/text.jsonl`. Reruns skip completed
traces and retry failed ones; failed requests are recorded, never dropped.

## Metrics

- **Who** (responsible-agent accuracy, multi-agent frameworks only), **When**
  (exact step accuracy), **What** (error-category macro-F1), **All** (joint) —
  the official per-framework-averaged recipe.
- Median and p95 **per-request latency**; total wall-clock is separate.
- Token usage and **estimated cost** (labeled estimates from published rates).
- 95% **confidence intervals** by cluster bootstrap over frameworks.
- **Calibration** (ECE) for any backend that returns probabilities (Jev).

## Protocol fidelity

This harness reuses the official `whowhen_eval` renderers, parser, and scorer
unchanged (pinned commit `14369dcb`). Deviations from a full official run:

1. **Subsampling.** We evaluate a fixed-seed stratified sample, not the whole
   6,257-trace subset, to bound cost. Aggregation still follows the official
   per-framework-mean recipe. Sampled ids are saved.
2. **Jev typed adaptation.** Described above; reported separately.

Because of (1) and (2), numbers here are **not comparable to the official
leaderboard** and are not presented as such.

## Dataset attribution

Who&When Pro, dataset `Leoxx/whowhen_pro`, licensed **CC-BY-4.0**. Paper:
Liu, Xi, Zhang, Zeng, Yue, Wang, Kang, Wu, Wang, *"Who&When Pro: Can LLMs Really
Attribute Failures in AI Agents?"*, arXiv:2607.09996 (2026). Harness:
[whowhenpro/whowhen_pro](https://github.com/whowhenpro/whowhen_pro).

## Pilots

Side experiments live under `pilots/`:

- [`pilots/gliclass-routerarena`](pilots/gliclass-routerarena) — a small pilot
  using the hosted GLiClass zero-shot classifier as an LLM router on RouterArena.
  Result on a 100-example sample: the router collapsed to one model and did not
  beat the best fixed model (61.5% vs 66.4%). Not a leaderboard result; not Jev.

## License

Code: Apache-2.0 (see `LICENSE`). The dataset keeps its own CC-BY-4.0 license and
is not included in this repository.

## Tests

```sh
pytest -q      # offline; no API keys required
```
