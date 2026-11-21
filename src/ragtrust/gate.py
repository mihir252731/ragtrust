"""Quality gates: absolute minimums and regression checks against a baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import HIGHER_IS_BETTER


@dataclass
class GateFailure:
    metric: str
    message: str


def parse_limits(spec: list[str] | None) -> dict[str, float]:
    """Parse ``["groundedness>=0.9", "injection_follow_rate<=0"]``-style limits.

    ``name=value`` is also accepted and uses the metric's natural direction.
    """
    limits: dict[str, float] = {}
    for item in spec or []:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            for op in (">=", "<=", "="):
                if op in part:
                    name, value = part.split(op, 1)
                    name = name.strip()
                    if name not in HIGHER_IS_BETTER:
                        raise ValueError(f"unknown metric {name!r}; choose from {sorted(HIGHER_IS_BETTER)}")
                    want_higher = HIGHER_IS_BETTER[name]
                    if (op == ">=" and not want_higher) or (op == "<=" and want_higher):
                        raise ValueError(f"{name}: lower is better" if not want_higher else f"{name}: higher is better")
                    limits[name] = float(value)
                    break
            else:
                raise ValueError(f"can't parse limit {part!r}; use metric>=value or metric<=value")
    return limits


def check_limits(summary: dict, limits: dict[str, float]) -> list[GateFailure]:
    failures = []
    for name, bound in limits.items():
        value = summary.get(name)
        if value is None:
            continue  # metric not measured on this dataset
        if HIGHER_IS_BETTER[name] and value < bound:
            failures.append(GateFailure(name, f"{name} = {value:.3f}, required >= {bound:.3f}"))
        if not HIGHER_IS_BETTER[name] and value > bound:
            failures.append(GateFailure(name, f"{name} = {value:.3f}, allowed <= {bound:.3f}"))
    return failures


def compare(baseline: dict, current: dict, max_drop: float = 0.02) -> tuple[list[dict], list[GateFailure]]:
    """Compare two summaries. A metric regresses if it worsens by more than ``max_drop``."""
    rows, failures = [], []
    for name, higher in HIGHER_IS_BETTER.items():
        b, c = baseline.get(name), current.get(name)
        if b is None or c is None:
            continue
        delta = c - b
        worse = -delta if higher else delta
        status = "regressed" if worse > max_drop + 1e-9 else ("improved" if worse < -1e-9 else "same")
        rows.append({"metric": name, "baseline": b, "current": c, "delta": round(delta, 4), "status": status})
        if status == "regressed":
            failures.append(GateFailure(name, f"{name} regressed {b:.3f} -> {c:.3f} (max drop {max_drop:.3f})"))
    return rows, failures
