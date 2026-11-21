"""ragtrust: measure whether a RAG system's answers can be trusted."""

__version__ = "0.1.0"

from .judges import CallableJudge, Judge, LexicalJudge  # noqa: E402
from .metrics import CaseResult, Summary, evaluate, evaluate_case  # noqa: E402
from .models import Case, Passage, load_cases  # noqa: E402

__all__ = [
    "Case",
    "CaseResult",
    "CallableJudge",
    "Judge",
    "LexicalJudge",
    "Passage",
    "Summary",
    "evaluate",
    "evaluate_case",
    "load_cases",
    "__version__",
]
