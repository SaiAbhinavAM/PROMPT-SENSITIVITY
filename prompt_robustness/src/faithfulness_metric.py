"""
faithfulness_metric.py — LLM-based faithfulness scoring (rectification R3).

Uses the independent 70B-AWQ instruct model (via llm_backend.ChatLLM) as an
NLI classifier: given the source text and a model response, the judge labels
the relationship as entailment / neutral / contradiction.

Faithfulness(source, response):
    entailment  → 1.0
    neutral     → 0.5
    contradiction → 0.0

Model: JUDGE_MODEL / FAITHFULNESS_MODEL env var
       (default: hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_LLM_MODEL = (
    os.getenv("FAITHFULNESS_MODEL")
    or os.getenv("JUDGE_MODEL")
    or "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
)

_LABEL_TO_SCORE = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}


def _faithfulness_llm_with_raw(responses: List[str], premise: str) -> Tuple[float, List[Dict]]:
    """LLM-as-NLI: returns (mean_score, per_variant_list) including raw labels."""
    from .llm_backend import get_chat_llm
    llm = get_chat_llm(_LLM_MODEL, quantization=os.getenv("JUDGE_QUANTIZATION"))
    valid = [(r or "").strip() for r in responses if (r or "").strip()]
    msgs = [
        [
            {"role": "system", "content": "You check summary faithfulness. Given a "
             "SOURCE and a SUMMARY, reply with exactly one word: entailment (fully "
             "supported), neutral (not contradicted but not stated), or contradiction."},
            {"role": "user", "content": f"SOURCE:\n{premise}\n\nSUMMARY:\n{r}\n\nLabel:"},
        ]
        for r in valid
    ]
    if not msgs:
        return 0.0, []
    outs = llm.chat_batch(msgs, max_new_tokens=4)
    per_variant: List[Dict] = []
    for raw in outs:
        low = raw.strip().lower()
        score = next((v for k, v in _LABEL_TO_SCORE.items() if k in low), 0.5)
        per_variant.append({"raw_label": raw.strip(), "score": round(score, 4)})
    mean = sum(d["score"] for d in per_variant) / len(per_variant) if per_variant else 0.0
    return mean, per_variant


def _faithfulness_llm(responses: List[str], premise: str) -> float:
    score, _ = _faithfulness_llm_with_raw(responses, premise)
    return score


def compute_faithfulness(
    responses: List[str],
    source_text: str,
    max_chars: int = 2000,
) -> Optional[float]:
    """Mean LLM faithfulness of each response given the source text.

    Returns mean score in [0, 1], or None if _LLM_MODEL is not configured.
    """
    premise = (source_text or "").strip()[:max_chars]
    if not premise:
        return None
    if not _LLM_MODEL:
        logger.error("No faithfulness model configured. Set FAITHFULNESS_MODEL or JUDGE_MODEL.")
        return None
    try:
        return _faithfulness_llm(responses, premise)
    except Exception as e:
        logger.error(f"Faithfulness scoring failed: {e}")
        return None


def compute_faithfulness_with_raw(
    responses: List[str],
    source_text: str,
    max_chars: int = 2000,
) -> Tuple[Optional[float], List[Dict]]:
    """Like compute_faithfulness but also returns per-variant raw outputs."""
    premise = (source_text or "").strip()[:max_chars]
    if not premise:
        return None, []
    if not _LLM_MODEL:
        logger.error("No faithfulness model configured. Set FAITHFULNESS_MODEL or JUDGE_MODEL.")
        return None, []
    try:
        return _faithfulness_llm_with_raw(responses, premise)
    except Exception as e:
        logger.error(f"Faithfulness scoring failed: {e}")
        return None, []
