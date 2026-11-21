"""Support judges: does a passage support a claim?

The default ``LexicalJudge`` is deterministic, offline and fast, which makes it
suitable as a CI gate. It is intentionally strict about numbers: a claim that
says "30 days" is never supported by a passage that says "14 days".

For higher recall on paraphrased answers, plug in any model through
``CallableJudge`` (see the README for an LLM example).
"""

from __future__ import annotations

from typing import Callable, Protocol

from .text import content_terms, numbers


class Judge(Protocol):
    name: str

    def support(self, claim: str, passage: str) -> float:
        """Return a support score in [0, 1]."""
        ...


class LexicalJudge:
    """Term-coverage judge with a hard number check.

    score = share of the claim's content terms found in the passage,
    or 0 if the claim contains a number the passage does not.
    """

    name = "lexical"

    def __init__(self, require_numbers: bool = True):
        self.require_numbers = require_numbers

    def support(self, claim: str, passage: str) -> float:
        claim_terms = content_terms(claim)
        if not claim_terms:
            return 1.0
        if self.require_numbers:
            claim_nums = numbers(claim)
            if claim_nums and not claim_nums <= numbers(passage):
                return 0.0
        passage_terms = content_terms(passage)
        return len(claim_terms & passage_terms) / len(claim_terms)


class CallableJudge:
    """Wrap any ``fn(claim, passage) -> float`` (e.g. an LLM or NLI model)."""

    def __init__(self, fn: Callable[[str, str], float], name: str = "custom"):
        self.fn = fn
        self.name = name

    def support(self, claim: str, passage: str) -> float:
        score = float(self.fn(claim, passage))
        return min(1.0, max(0.0, score))
