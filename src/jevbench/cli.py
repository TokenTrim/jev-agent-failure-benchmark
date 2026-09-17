"""jevbench command line: sample, estimate, run (Jev), report."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from whowhen_eval.prompts import load_taxonomy, parts_to_text
from whowhen_eval.run import build_prompt

from .dataset import Example, load_examples, load_sample, release_of, save_sample, stratified_sample
from .pricing import cost_usd


def _examples_for(text: Path, sample_path: Path) -> list[Example]:
    ids = set(load_sample(sample_path))
    picked = [ex for ex in load_examples(text) if ex.id in ids]
    missing = ids - {ex.id for ex in picked}
    if missing:
        raise SystemExit(f"{len(missing)} sampled ids not found in {text} (revision mismatch?)")
    return picked


def cmd_sample(args: argparse.Namespace) -> int:
    ids = stratified_sample(load_examples(args.text), args.n, args.seed, floor=args.floor)
    save_sample(args.out, ids, seed=args.seed, n=args.n)
    print(f"sampled {len(ids)} ids (seed={args.seed}) -> {args.out}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    """Offline: render each sampled trace, estimate input tokens and Jev cost."""
    taxonomy = load_taxonomy(args.data_root)
    examples = _examples_for(args.text, args.sample)
    in_tokens = sum(
        len(parts_to_text(build_prompt(release_of(ex, args.data_root), ex.framework, taxonomy))) // 4
        for ex in examples
    )
    print(f"traces: {len(examples)}  est. input tokens: {in_tokens:,}  "
          f"est. Jev cost: ${cost_usd(args.jev_model, in_tokens, 0):.2f} (output free)")
    print("Estimate only (~4 chars/token). Jev bills input tokens at the published rate.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .backends.jev import JevBackend
    from .runner import run_backend

    taxonomy = load_taxonomy(args.data_root)
    examples = _examples_for(args.text, args.sample)
    backend = JevBackend(args.jev_model)
    out_path = args.run_dir / "jev" / "text.jsonl"
    print(f"jev model={backend.model} traces={len(examples)} -> {out_path}")
    counts = asyncio.run(run_backend(
        backend, examples, taxonomy, args.data_root, out_path, concurrency=args.concurrency,
    ))
    print(f"completed={counts['completed']} skipped={counts['skipped']} errored={counts['errored']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report_vs_paper as rp

    results = args.results or args.run_dir / "jev" / "text.jsonl"
    row = rp.jev_row(results)
    md = rp.render_markdown(row)
    args.out.write_text(md, encoding="utf-8")
    rp.render_chart(row, args.chart)
    print(md)
    print(f"wrote {args.out} and {args.chart}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they work after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=Path("data"),
                        help="dataset checkout root (holds taxonomy.yaml and data/)")
    common.add_argument("--text", type=Path, default=Path("data/text.jsonl"))
    common.add_argument("--run-dir", type=Path, default=Path("results/run"))
    common.add_argument("--jev-model", default="jev-1.13.0")

    p = argparse.ArgumentParser(prog="jevbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", parents=[common], help="draw a fixed-seed stratified sample")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=20240517)
    s.add_argument("--floor", type=int, default=20)
    s.add_argument("--out", type=Path, default=Path("results/run/sample.json"))
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("estimate", parents=[common], help="offline input-token/cost estimate")
    e.add_argument("--sample", type=Path, default=Path("results/run/sample.json"))
    e.set_defaults(func=cmd_estimate)

    r = sub.add_parser("run", parents=[common], help="run Jev over a sample (resumable)")
    r.add_argument("--sample", type=Path, default=Path("results/run/sample.json"))
    r.add_argument("--concurrency", type=int, default=16)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", parents=[common], help="metrics, paper comparison, chart")
    rp.add_argument("--results", type=Path, default=None)
    rp.add_argument("--out", type=Path, default=Path("RESULTS.md"))
    rp.add_argument("--chart", type=Path, default=Path("figures/whowhen_jev_vs_llm.png"))
    rp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
