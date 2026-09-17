# Who&When Pro — Jev vs frontier LLMs (text subset)

| Model | Who | When | What | All |
|---|---|---|---|---|
| Jev (task framing) (ours, n=6257) | **73.4** [70.5, 76.2] | **76.4** [74.3, 78.4] | **23.7** [22.4, 24.9] | **31.3** [29.5, 33.0] |
| GPT-5.4 *(paper)* | 55.7 | 72.3 | 15.3 | 21.3 |
| Claude Sonnet 4.6 *(paper)* | 54.9 | 69.8 | 19.1 | 22.4 |
| GLM-5 *(paper)* | 54.9 | 71.1 | 22.2 | 25.3 |
| Qwen3.5-122B *(paper)* | 57.5 | 73.9 | 17.0 | 21.6 |

Jev cost: **$1.28** for all 6257 traces (30,497,481 input tokens at $0.042/Mtok; output free). Latency is not compared here (the endpoint is a shared early-access service; use Jev's published per-token throughput instead of our queue-affected timings).

**Read this carefully.** Who and When are **adaptation-favoured**: Jev's typed API is handed the trace's enumerated agent ids and step coordinates and picks one, while the paper's LLMs free-generate the answer. The like-for-like axis is **What** (error macro-F1) where both sides classify over the identical 17-code taxonomy: Jev 23.7 vs the best LLM 22.2. Mode-confidence calibration ECE: 0.287. Baselines: arXiv:2607.09996 Table 4. Not a leaderboard submission.

## Chart

![Jev vs frontier LLMs](figures/whowhen_jev_vs_llm.png)

## Two prompt configurations (transparency)

We report two frozen Jev configs. Neither was tuned against Who&When scores.

| Config | Who | When | What | All |
|---|---|---|---|---|
| minimal instructions | 69.8 | 75.6 | 22.1 | 28.9 |
| **task framing (headline)** | **73.4** | **76.4** | **23.7** | **31.3** |

"Task framing" adds one sentence to Jev's state: the same definition of "the first
decisive error" that the official LLM prompt already puts in its header. It is a
fairness fix (parity with the baselines' prompt), designed a priori, applied once,
and re-run on the full set. It is the headline config. `minimal` is the terser
first version, kept for transparency.
