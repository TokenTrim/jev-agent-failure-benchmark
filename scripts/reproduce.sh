#!/usr/bin/env sh
# Reproduce the benchmark end to end. Needs TYPESAFE_API_KEY in the environment.
set -eu

SEED=20240517
N=6257                       # whole text subset; use a smaller N for a subset
REV=0bd196c8a040841c4ae167ab33cc8151de246f1f

# 1. Dataset (pinned revision), if not already present.
if [ ! -f data/text.jsonl ]; then
  mkdir -p data
  curl -L "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/$REV/data/text.jsonl" -o data/text.jsonl
  curl -L "https://huggingface.co/datasets/Leoxx/whowhen_pro/resolve/$REV/taxonomy.yaml" -o data/taxonomy.yaml
fi

# 2. Fixed-seed sample, offline cost estimate, Jev run, report.
jevbench sample --n "$N" --seed "$SEED" --out results/run/sample.json
jevbench estimate --sample results/run/sample.json
jevbench run --sample results/run/sample.json --concurrency 16
jevbench report
