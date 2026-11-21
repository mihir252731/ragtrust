"""Command-line interface.

    ragtrust run cases.jsonl --html report.html --json results.json \
        --require "groundedness>=0.9,injection_follow_rate<=0"
    ragtrust compare baseline.json results.json --max-drop 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .gate import check_limits, compare, parse_limits
from .judges import LexicalJudge
from .metrics import DEFAULT_SUPPORT_THRESHOLD, evaluate
from .models import load_cases
from .report import LABELS, render_html, write_html, write_json


def _fmt(v):
    return "n/a" if v is None else f"{v * 100:6.1f}%"


def _print_summary(summary: dict) -> None:
    width = max(len(label) for label, _ in LABELS.values())
    for name, (label, _) in LABELS.items():
        print(f"  {label:<{width}}  {_fmt(summary.get(name))}")


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    results, summary = evaluate(cases, LexicalJudge(), args.threshold)
    s = summary.to_dict()

    baseline_rows, regressions = None, []
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["summary"]
        baseline_rows, regressions = compare(base, s, args.max_drop)

    limits = parse_limits(args.require)
    failures = check_limits(s, limits) + regressions
    gate_msgs = [f.message for f in failures] if (limits or args.baseline) else None

    print(f"ragtrust {__version__}: {len(cases)} cases from {args.cases}")
    _print_summary(s)
    if args.json:
        write_json(args.json, results, summary, str(args.cases))
        print(f"  wrote {args.json}")
    if args.html:
        write_html(args.html, render_html(results, summary, str(args.cases), baseline_rows, gate_msgs))
        print(f"  wrote {args.html}")
    if failures:
        print("\nQuality gate FAILED:")
        for f in failures:
            print(f"  - {f.message}")
        return 1
    if gate_msgs is not None:
        print("\nQuality gate passed.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["summary"]
    cur = json.loads(Path(args.current).read_text(encoding="utf-8"))["summary"]
    rows, failures = compare(base, cur, args.max_drop)
    print(f"{'metric':<28}{'baseline':>10}{'current':>10}{'change':>10}  status")
    for r in rows:
        print(
            f"{LABELS[r['metric']][0]:<28}{_fmt(r['baseline']):>10}{_fmt(r['current']):>10}"
            f"{r['delta'] * 100:>+8.1f}pt  {r['status']}"
        )
    if failures:
        print("\nRegressions:")
        for f in failures:
            print(f"  - {f.message}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragtrust", description="Measure whether RAG answers can be trusted.")
    p.add_argument("--version", action="version", version=f"ragtrust {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="evaluate a JSONL file of cases")
    r.add_argument("cases", type=Path)
    r.add_argument("--json", type=Path, help="write per-case results and summary as JSON")
    r.add_argument("--html", type=Path, help="write a self-contained HTML report")
    r.add_argument("--threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD,
                   help="support score needed to count a sentence or citation as supported (default 0.6)")
    r.add_argument("--require", action="append", metavar="LIMITS",
                   help='minimums, e.g. "groundedness>=0.9,injection_follow_rate<=0" (repeatable)')
    r.add_argument("--baseline", type=Path, help="results JSON to compare against; regressions fail the run")
    r.add_argument("--max-drop", type=float, default=0.02, help="allowed worsening vs baseline (default 0.02)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="compare two results JSON files")
    c.add_argument("baseline", type=Path)
    c.add_argument("current", type=Path)
    c.add_argument("--max-drop", type=float, default=0.02)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ragtrust: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
