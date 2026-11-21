"""Small, dependency-free text utilities."""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its just may might more most must no not of on or our out
    so such than that the their them then there these they this those to too up us
    was we were what when where which while who why will with would you your yes also
    any all each per via about after before over under again only very own same both
    should shall get gets got""".split()
)

ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't|cannot|can't) (?:know|find|answer|say)\b",
    r"\b(?:not|isn't|is not) (?:enough|sufficient) (?:information|context|evidence)\b",
    r"\b(?:no|couldn't find|could not find) (?:relevant |supporting )?(?:information|documentation|sources?|evidence)\b",
    r"\bunable to (?:answer|find|determine)\b",
    r"\b(?:the )?(?:provided |available )?(?:documents|documentation|docs|context|sources|knowledge base) (?:do not|don't|does not|doesn't) (?:say|mention|cover|contain|specify)\b",
    r"\bplease contact (?:support|your administrator)\b.*\b(?:not|no)\b",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, keeping citation markers attached.

    Handles decimals ("2.4 s"), common abbreviations and bullet lists.
    """
    text = text.replace("\r", "")
    pieces: list[str] = []
    for block in re.split(r"\n\s*(?:[-*•]|\d+[.)])\s+|\n{2,}", text):
        block = " ".join(block.split())
        if not block:
            continue
        # Split after . ! ? (optionally followed by citations) when followed
        # by whitespace and an uppercase letter, digit or quote.
        parts = re.split(r"(?<=[.!?])((?:\s*\[[^\]]+\])*)\s+(?=[A-Z0-9\"'(])", block)
        # re.split with a group returns [sent, cites, sent, cites, ...]
        buf = ""
        for i, part in enumerate(parts):
            if part is None:
                continue
            if i % 2 == 0:
                buf = part
            else:
                pieces.append((buf + part).strip())
                buf = ""
        if buf.strip():
            pieces.append(buf.strip())
    return [p for p in pieces if _content_tokens(strip_citations(p))]


def extract_citations(sentence: str) -> list[str]:
    seen: list[str] = []
    for cid in CITATION_RE.findall(sentence):
        if cid not in seen:
            seen.append(cid)
    return seen


def strip_citations(sentence: str) -> str:
    return " ".join(CITATION_RE.sub(" ", sentence).split())


def _stem(tok: str) -> str:
    # Deliberately tiny stemmer: enough to match "retries"/"retry",
    # "configured"/"configure", "limits"/"limit".
    if tok[0].isdigit():
        return tok
    for suf in ("ations", "ation", "ingly", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            base = tok[: -len(suf)]
            if suf in ("ies", "ied"):
                base += "y"
            return base
    return tok


def _content_tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    """Stemmed content words, with numbers kept verbatim."""
    return {_stem(t) for t in _content_tokens(text)}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def is_abstention(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    return bool(_ABSTAIN_RE.search(stripped))
