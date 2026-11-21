"""Data model for evaluation cases.

A case is one question sent to a RAG system, plus what the system retrieved
and what it answered. Cases are stored one per line in JSONL files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Passage:
    """A retrieved chunk of text."""

    id: str
    text: str
    # Groups allowed to read this passage. None means "public".
    acl: list[str] | None = None
    # If this passage contains a planted prompt injection, the secret string
    # the injection tries to make the model print. Used to test resistance.
    injection_canary: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Passage":
        return cls(
            id=str(d["id"]),
            text=str(d["text"]),
            acl=list(d["acl"]) if d.get("acl") is not None else None,
            injection_canary=d.get("injection_canary"),
        )

    def readable_by(self, groups: Iterable[str] | None) -> bool:
        if self.acl is None:
            return True
        if groups is None:
            return False
        return bool(set(self.acl) & set(groups))


@dataclass
class Case:
    """One evaluation case."""

    id: str
    question: str
    answer: str
    retrieved: list[Passage]
    # False when the knowledge base cannot answer the question; the correct
    # behaviour is then to abstain ("I don't know").
    answerable: bool = True
    # Passage ids a perfect retriever would return. Optional.
    gold_passage_ids: list[str] = field(default_factory=list)
    # Groups of the user who asked. None means permissions are not tested.
    user_groups: list[str] | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        missing = [k for k in ("id", "question", "answer", "retrieved") if k not in d]
        if missing:
            raise ValueError(f"case {d.get('id', '?')} is missing fields: {', '.join(missing)}")
        return cls(
            id=str(d["id"]),
            question=str(d["question"]),
            answer=str(d["answer"]),
            retrieved=[Passage.from_dict(p) for p in d["retrieved"]],
            answerable=bool(d.get("answerable", True)),
            gold_passage_ids=[str(x) for x in d.get("gold_passage_ids", [])],
            user_groups=list(d["user_groups"]) if d.get("user_groups") is not None else None,
            tags=list(d.get("tags", [])),
        )

    def passage(self, pid: str) -> Passage | None:
        for p in self.retrieved:
            if p.id == pid:
                return p
        return None


def load_cases(path: str | Path) -> list[Case]:
    """Load cases from a JSONL file (one JSON object per line)."""
    return list(iter_cases(path))


def iter_cases(path: str | Path) -> Iterator[Case]:
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = Case.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
            seen.add(case.id)
            yield case
