"""Use an LLM as the support judge instead of the offline lexical judge.

The lexical judge is strict and deterministic, which is what you want in CI.
An LLM judge catches paraphrases the lexical judge misses. Running both and
reviewing disagreements is a good way to tune your support threshold.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python examples/llm_judge.py examples/support_kb_v2.jsonl
"""

import sys

from ragtrust import CallableJudge, evaluate, load_cases

PROMPT = """You check whether a source passage supports a claim.
Answer with a single number from 0 to 1: 1 = the passage fully supports the claim,
0 = it does not support it or contradicts it. Numbers in the claim must match the passage.

Passage:
{passage}

Claim:
{claim}

Score:"""


def make_llm_judge(model: str = "claude-haiku-5-5"):
    import anthropic

    client = anthropic.Anthropic()

    def score(claim: str, passage: str) -> float:
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": PROMPT.format(passage=passage, claim=claim)}],
        )
        try:
            return float(msg.content[0].text.strip())
        except ValueError:
            return 0.0

    return CallableJudge(score, name=f"llm:{model}")


if __name__ == "__main__":
    results, summary = evaluate(load_cases(sys.argv[1]), judge=make_llm_judge())
    for key, value in summary.to_dict().items():
        print(f"{key:28} {value}")
