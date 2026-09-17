# Who&When Pro - Jev vs frontier LLMs (text subset)

| Model | Who | When | What | All |
|---|---|---|---|---|
| Jev (ours, n=6257) | **73.4** [70.5, 76.2] | **76.4** [74.3, 78.4] | **23.7** [22.4, 24.9] | **31.3** [29.5, 33.0] |
| GPT-5.4 *(paper)* | 55.7 | 72.3 | 15.3 | 21.3 |
| Claude Sonnet 4.6 *(paper)* | 54.9 | 69.8 | 19.1 | 22.4 |
| GLM-5 *(paper)* | 54.9 | 71.1 | 22.2 | 25.3 |
| Qwen3.5-122B *(paper)* | 57.5 | 73.9 | 17.0 | 21.6 |

Jev cost: **$1.28** for all 6257 traces (30,497,481 input tokens at $0.042/Mtok; output free).

**Who and When are adaptation-favoured**: Jev picks the agent/step from the trace's enumerated options, while the paper's LLMs free-generate. The like-for-like axis is **What** (error macro-F1, same 17-code taxonomy for both): Jev 23.7 vs the best LLM 22.2. Mode-confidence ECE 0.287. Latency is not compared (shared early-access endpoint). Baselines: arXiv:2607.09996 Table 4. Failures in Who&When Pro are injected, not natural incidents.
