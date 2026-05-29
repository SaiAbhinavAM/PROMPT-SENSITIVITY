import os
import re
from typing import Dict, List, Tuple

# Judge model — always the independent 70B-AWQ instruct model on GPU.
# Override via JUDGE_MODEL env var.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")


def _causal_judge_texts(input_text: str, summaries: list) -> List[str]:
    """Get raw text outputs from the causal 70B judge (batched)."""
    from .llm_backend import get_chat_llm
    llm = get_chat_llm(JUDGE_MODEL, quantization=os.getenv("JUDGE_QUANTIZATION"))
    msgs = [
        [
            {"role": "system", "content": "You are a strict summary evaluator. "
             "Rate how well the summary captures the article's main facts on a 1-5 "
             "scale (1=very poor, 5=excellent). Reply with ONLY the single digit."},
            {"role": "user", "content": f"Article:\n{input_text[:4000]}\n\nSummary:\n{s}\n\nScore (1-5):"},
        ]
        for s in summaries
    ]
    return llm.chat_batch(msgs, max_new_tokens=4)


def parse_judge_output(text: str) -> float:
    match = re.search(r"\d", text)
    if match:
        return int(match.group()) / 5.0
    return 0.5


def llm_judge_mean(input_text: str, summaries: list) -> float:
    """Judge ALL prompt variants and return the mean normalized score in [0, 1]."""
    mean, _ = llm_judge_with_raw(input_text, summaries)
    return mean


def llm_judge_with_raw(
    input_text: str, summaries: list
) -> Tuple[float, List[Dict]]:
    """Judge ALL variants; return (mean_score, per_variant_raw_outputs).

    per_variant_raw_outputs — one dict per valid variant:
      {"raw_output": str, "score": float}
    """
    valid = [s for s in summaries if (s or "").strip()]
    if not valid:
        return 0.5, []

    texts = _causal_judge_texts(input_text, valid)
    raw_outputs: List[Dict] = []
    for raw_text in texts:
        score = parse_judge_output(raw_text)
        raw_outputs.append({"raw_output": raw_text.strip(), "score": round(score, 4)})

    scores = [d["score"] for d in raw_outputs]
    mean_score = sum(scores) / len(scores) if scores else 0.5
    return mean_score, raw_outputs
