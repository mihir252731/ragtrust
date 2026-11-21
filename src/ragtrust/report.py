"""JSON and self-contained HTML reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from . import __version__
from .metrics import HIGHER_IS_BETTER, CaseResult, Summary

LABELS = {
    "pass_rate": ("Cases passing", "Share of cases with no issues"),
    "groundedness": ("Groundedness", "Answer sentences supported by a retrieved source"),
    "unsupported_citation_rate": ("Unsupported citations", "Citations whose source doesn't back the sentence"),
    "citation_coverage": ("Citation coverage", "Answer sentences that carry a citation"),
    "abstention_accuracy": ("Correct abstentions", "Unanswerable questions where the system said it doesn't know"),
    "false_abstention_rate": ("False abstentions", "Answerable questions the system refused"),
    "retrieval_recall": ("Retrieval recall", "Gold passages found by the retriever"),
    "permission_leak_rate": ("Permission leaks", "Cases that retrieved or cited a passage the user can't read"),
    "injection_follow_rate": ("Injection followed", "Planted prompt injections the answer obeyed"),
}


def write_json(path: str | Path, results: list[CaseResult], summary: Summary, dataset: str) -> None:
    payload = {
        "tool": "ragtrust",
        "version": __version__,
        "dataset": dataset,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary.to_dict(),
        "cases": [r.to_dict() for r in results],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _tone(name: str, v: float | None) -> str:
    if v is None:
        return "muted"
    if name in ("permission_leak_rate", "injection_follow_rate"):
        return "good" if v == 0 else "bad"  # zero tolerance
    good = v if HIGHER_IS_BETTER[name] else 1 - v
    return "good" if good >= 0.9 else ("warn" if good >= 0.7 else "bad")


CSS = """
:root{--bg:#fbfaf7;--card:#fff;--ink:#1d1f24;--muted:#6b6f76;--line:#e6e3dc;
--good:#1f7a4d;--good-bg:#e7f4ec;--warn:#9a6200;--warn-bg:#fbf1dc;--bad:#b3261e;--bad-bg:#fbe9e7;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 24px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:18px;margin:40px 0 12px}
.sub{color:var(--muted);margin:0 0 28px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.card .k{font-weight:600;font-size:13px}.card .d{color:var(--muted);font-size:12.5px}
.good .v{color:var(--good)}.warn .v{color:var(--warn)}.bad .v{color:var(--bad)}.muted .v{color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#f6f4ef}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.pill.good{background:var(--good-bg);color:var(--good)}.pill.bad{background:var(--bad-bg);color:var(--bad)}
.pill.warn{background:var(--warn-bg);color:var(--warn)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:10px 14px}
summary{cursor:pointer;font-weight:600}summary .q{font-weight:400;color:var(--muted)}
ul.issues{margin:8px 0 20px;padding-left:20px;color:var(--bad)}
.s{padding:6px 10px;border-left:3px solid;margin:6px 0;border-radius:0 6px 6px 0;font-size:14px}
.s.ok{border-color:var(--good);background:var(--good-bg)}.s.no{border-color:var(--bad);background:var(--bad-bg)}
.s small{color:var(--muted);display:block}
footer{margin-top:48px;color:var(--muted);font-size:12.5px}
"""


def render_html(
    results: list[CaseResult],
    summary: Summary,
    dataset: str,
    comparison: list[dict] | None = None,
    gate_messages: list[str] | None = None,
) -> str:
    s = summary.to_dict()
    cards = []
    for name, (label, desc) in LABELS.items():
        v = s.get(name)
        cards.append(
            f'<div class="card {_tone(name, v)}"><div class="k">{label}</div>'
            f'<div class="v">{_pct(v)}</div><div class="d">{desc}</div></div>'
        )

    gate_html = ""
    if gate_messages is not None:
        if gate_messages:
            items = "".join(f"<li>{escape(m)}</li>" for m in gate_messages)
            gate_html = f'<h2>Quality gate: <span class="pill bad">FAILED</span></h2><ul class="issues">{items}</ul>'
        else:
            gate_html = '<h2>Quality gate: <span class="pill good">PASSED</span></h2>'

    comp_html = ""
    if comparison:
        rows = "".join(
            f"<tr><td>{LABELS[r['metric']][0]}</td><td class='n'>{_pct(r['baseline'])}</td>"
            f"<td class='n'>{_pct(r['current'])}</td><td class='n'>{r['delta'] * 100:+.1f} pts</td>"
            f"<td><span class='pill {'bad' if r['status'] == 'regressed' else 'good' if r['status'] == 'improved' else 'warn'}'>"
            f"{r['status']}</span></td></tr>"
            for r in comparison
        )
        comp_html = (
            "<h2>Compared with baseline</h2><table><tr><th>Metric</th><th>Baseline</th>"
            f"<th>Current</th><th>Change</th><th>Status</th></tr>{rows}</table>"
        )

    failing = [r for r in results if not r.passed]
    case_blocks = []
    for r in failing:
        issues = "".join(f"<li>{escape(i)}</li>" for i in r.issues)
        sents = "".join(
            f'<div class="s {"ok" if x.supported else "no"}">{escape(x.text)}'
            f"<small>best support {x.best_support:.2f} from {escape(str(x.best_passage))}"
            + (f"; cited {', '.join(f'{escape(k)}={v:.2f}' for k, v in x.cited_support.items())}" if x.cited_support else "")
            + "</small></div>"
            for x in r.sentences
        )
        tags = " ".join(f'<span class="pill warn">{escape(t)}</span>' for t in r.tags)
        case_blocks.append(
            f"<details><summary>{escape(r.case_id)} {tags} <span class='q'>{escape(r.question)}</span></summary>"
            f"<ul class='issues'>{issues}</ul>{sents}</details>"
        )
    cases_html = "".join(case_blocks) or "<p>Every case passed.</p>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ragtrust report: {escape(dataset)}</title><style>{CSS}</style></head>
<body><main>
<h1>RAG trust report</h1>
<p class="sub">{escape(dataset)} · {summary.cases} cases · judge: {escape(summary.judge)} · support threshold {summary.threshold:.2f}</p>
{gate_html}
<div class="grid">{''.join(cards)}</div>
{comp_html}
<h2>Cases with issues ({len(failing)} of {len(results)})</h2>
{cases_html}
<footer>Generated by ragtrust {__version__} on {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}.</footer>
</main></body></html>"""


def write_html(path: str | Path, html: str) -> None:
    Path(path).write_text(html, encoding="utf-8")
