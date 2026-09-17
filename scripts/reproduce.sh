#!/usr/bin/env sh
# Reproduce the benchmark. Canonical settings live here.
set -eu

SEED=20240517
N=300
LLM_MODEL=${LLM_MODEL:-gpt-5.6-terra}
JEV_MODEL=${JEV_MODEL:-jev-1.13.0}
RUN_DIR=results/run
REV=0bd196c8a040841c4ae167ab33cc8151de246f1f

# 1. Dataset (pinned revision), if not already present.
if [ ! -f data/text.jsonl ]; then
  mkdir -p data
  curl -L "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/$REV/data/text.jsonl" -o data/text.jsonl
  curl -L "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/$REV/taxonomy.yaml" -o data/taxonomy.yaml
fi

# 2. Fixed-seed, framework-stratified sample.
jevbench sample --n "$N" --seed "$SEED" --out "$RUN_DIR/sample.json"

# 3. Offline cost estimate (no API calls).
jevbench estimate --sample "$RUN_DIR/sample.json" --llm-model "$LLM_MODEL"

# 4. Both backends (resumable). Needs TYPESAFE_API_KEY and the baseline's key.
jevbench --run-dir "$RUN_DIR" run --sample "$RUN_DIR/sample.json" \
  --backend jev --backend llm --jev-model "$JEV_MODEL" --llm-model "$LLM_MODEL" --concurrency 8

# 5. Report + chart.
jevbench --run-dir "$RUN_DIR" report --llm-model "$LLM_MODEL" --jev-model "$JEV_MODEL" \
  --chart "$RUN_DIR/chart.png"
