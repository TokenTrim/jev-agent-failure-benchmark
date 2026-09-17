"""jevbench command line: sample, estimate, run, report."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from whowhen_eval.prompts import load_taxonomy, parts_to_text
from whowhen_eval.run import build_prompt

from . import report as report_mod
from .dataset import Example, load_examples, load_sample, release_of, save_sample, stratified_sample
from .pricing import cost_usd, price_for


def _examples_for(text: Path, sample_path: Path | None) -> list[Example]:
    examples = load_examples(text)
    if sample_path is None:
        return examples
    ids = set(load_sample(sample_path))
    picked = [ex for ex in examples if ex.id in ids]
    missing = ids - {ex.id for ex in picked}
    if missing:
        raise SystemExit(f"{len(missing)} sampled ids not found in {text} (revision mismatch?)")
    return picked


def cmd_sample(args: argparse.Namespace) -> int:
    examples = load_examples(args.text)
    ids = stratified_sample(examples, args.n, args.seed, floor=args.floor)
    save_sample(args.out, ids, seed=args.seed, n=args.n)
    print(f"sampled {len(ids)} ids (seed={args.seed}) -> {args.out}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    """Offline: render each sampled trace, estimate tokens, price it. No API calls."""
    taxonomy = load_taxonomy(args.data_root)
    examples = _examples_for(args.text, args.sample)
    in_tokens = 0
    for ex in examples:
        parts = build_prompt(release_of(ex, args.data_root), ex.framework, taxonomy)
        in_tokens += int(len(parts_to_text(parts)) / 4)  # ~4 chars/token, labeled estimate
    print(f"traces: {len(examples)}  est. input tokens (shared): {in_tokens:,}")
    print(f"assumed output tokens/trace (LLM): {args.out_tokens}")
    for model in [args.jev_model, args.llm_model]:
        out = 0 if model == args.jev_model else args.out_tokens * len(examples)
        c = cost_usd(model, in_tokens, out)
        priced = "" if price_for(model) else "  (UNPRICED — add to pricing.py)"
        print(f"  {model:16s} in={in_tokens:>10,} out={out:>10,} est=${(c or 0):.2f}{priced}")
    print("Estimates from published rates, not billed amounts. ~4 chars/token heuristic.")
    return 0


def _build_backend(name: str, args: argparse.Namespace):
    if name == "jev":
        from .backends.jev import JevBackend
        return JevBackend(args.jev_model)
    if name == "llm":
        from .backends.llm import LLMBackend
        return LLMBackend(args.llm_model, reasoning_effort=args.reasoning_effort)
    raise SystemExit(f"unknown backend: {name}")


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_backend

    taxonomy = load_taxonomy(args.data_root)
    examples = _examples_for(args.text, args.sample)
    for name in args.backend:
        backend = _build_backend(name, args)
        out_path = args.run_dir / name / "text.jsonl"
        print(f"[{name}] model={backend.model} traces={len(examples)} -> {out_path}")
        counts = asyncio.run(run_backend(
            backend, examples, taxonomy, args.data_root, out_path, concurrency=args.concurrency,
        ))
        print(f"[{name}] completed={counts['completed']} skipped={counts['skipped']} "
              f"errored={counts['errored']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    backends = {"jev": args.jev_model, "llm": args.llm_model}
    report = report_mod.write_report(args.run_dir, backends)
    print(report_mod.render_markdown(report))
    if args.chart:
        report_mod.render_chart(report, args.chart)
        print(f"chart -> {args.chart}")
    print(f"results.json + REPORT.md -> {args.run_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    # Common options live on a parent parser so they work after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=Path("data"),
                        help="dataset checkout root (holds taxonomy.yaml and data/)")
    common.add_argument("--text", type=Path, default=Path("data/text.jsonl"))
    common.add_argument("--run-dir", type=Path, default=Path("results/run"))
    common.add_argument("--jev-model", default="jev-1.13.0")
    common.add_argument("--llm-model", default="gpt-5.6-terra")

    p = argparse.ArgumentParser(prog="jevbench", description=__doc__, parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw and save a stratified sample manifest",
                       parents=[common])
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=20240517)
    s.add_argument("--floor", type=int, default=20)
    s.add_argument("--out", type=Path, default=Path("results/run/sample.json"))
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("estimate", help="offline token/cost estimate for a sample", parents=[common])
    e.add_argument("--sample", type=Path, default=Path("results/run/sample.json"))
    e.add_argument("--out-tokens", type=int, default=1500, help="assumed LLM output tokens/trace")
    e.set_defaults(func=cmd_estimate)

    r = sub.add_parser("run", help="run one or both backends over a sample", parents=[common])
    r.add_argument("--sample", type=Path, default=Path("results/run/sample.json"))
    r.add_argument("--backend", action="append", choices=["jev", "llm"], required=True)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--reasoning-effort", default=None)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="build results.json, REPORT.md and chart", parents=[common])
    rp.add_argument("--chart", type=Path, default=None)
    rp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
