"""
faithfulness_metric.py — NLI-based faithfulness scoring (rectification R3).

Replaces the bag-of-words Hallucination Score (HS) as the faithfulness axis of
PRI with a proper Natural Language Inference check: a response is faithful to
its source if the source *entails* (or at least does not *contradict*) the
response.

Model: cross-encoder/nli-deberta-v3-small (~140M params, < 2B).
       Label order: {0: contradiction, 1: entailment, 2: neutral}.

Faithfulness(source, response) = P(entailment) + 0.5 * P(neutral)
    → 1.0 when the source clearly supports the response
    → 0.0 when the source contradicts it

If the NLI model cannot be loaded (offline / download failure), the caller is
expected to fall back to the legacy (1 - HS) signal. `nli_available()` reports
whether the model is usable.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_model = None
_load_failed = False

# Backend: "nli" (default, independent deberta cross-encoder) or "llm"
# (LLM-as-NLI using a large independent instruct model — H100, no-<2B policy).
_BACKEND = os.getenv("FAITHFULNESS_BACKEND", "nli").lower()
_LLM_MODEL = os.getenv("FAITHFULNESS_MODEL") or os.getenv("JUDGE_MODEL", "")


def _get_model():
    """Lazily load the cross-encoder NLI model; cache load failures."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading NLI faithfulness model: {_MODEL_NAME}")
        _model = CrossEncoder(_MODEL_NAME)
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning(f"NLI model unavailable ({e}); faithfulness will fall back to 1 - HS")
        _load_failed = True
        _model = None
    return _model


def nli_available() -> bool:
    """True if the NLI model is loadable in this environment."""
    return _get_model() is not None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def compute_faithfulness(
    responses: List[str],
    source_text: str,
    max_chars: int = 2000,
) -> Optional[float]:
    """Mean NLI faithfulness of each response given the source text.

    Args:
        responses:   model outputs (one per prompt variant).
        source_text: the grounding text the outputs must stay faithful to.
        max_chars:   truncate the premise to keep it within the encoder window.

    Returns:
        Mean faithfulness in [0, 1], or None if the NLI model is unavailable
        (so the caller can fall back to a different signal).
    """
    premise = (source_text or "").strip()[:max_chars]
    if not premise:
        return None

    # LLM-as-NLI backend (large independent instruct model).
    if _BACKEND == "llm" and _LLM_MODEL:
        try:
            return _faithfulness_llm(responses, premise)
        except Exception as e:  # pragma: no cover - H100 path
            logger.warning(f"LLM faithfulness failed ({e}); falling back to NLI/1-HS")

    model = _get_model()
    if model is None:
        return None

    pairs = [(premise, (r or "").strip()) for r in responses if (r or "").strip()]
    if not pairs:
        return 0.0

    logits = model.predict(pairs)
    logits = np.atleast_2d(np.asarray(logits, dtype=float))

    scores = []
    for row in logits:
        probs = _softmax(row)  # [contradiction, entailment, neutral]
        faith = float(probs[1] + 0.5 * probs[2])
        scores.append(max(0.0, min(1.0, faith)))

    return float(np.mean(scores)) if scores else 0.0


def compute_faithfulness_with_raw(
    responses: List[str],
    source_text: str,
    max_chars: int = 2000,
) -> Tuple[Optional[float], List[Dict]]:
    """Like compute_faithfulness but also returns per-variant raw NLI outputs.

    Returns (mean_score, per_variant_list).
    NLI backend  → each dict: {"entail_prob": float, "neutral_prob": float, "score": float}
    LLM-as-NLI  → each dict: {"raw_label": str, "score": float}
    Returns (None, []) if neither backend is available.
    """
    premise = (source_text or "").strip()[:max_chars]
    if not premise:
        return None, []

    if _BACKEND == "llm" and _LLM_MODEL:
        try:
            score, per_variant = _faithfulness_llm_with_raw(responses, premise)
            return score, per_variant
        except Exception as e:
            logger.warning(f"LLM faithfulness failed ({e}); falling back to NLI")

    model = _get_model()
    if model is None:
        return None, []

    pairs = [(premise, (r or "").strip()) for r in responses if (r or "").strip()]
    if not pairs:
        return 0.0, []

    logits = model.predict(pairs)
    logits = np.atleast_2d(np.asarray(logits, dtype=float))

    per_variant: List[Dict] = []
    for row in logits:
        probs = _softmax(row)          # [contradiction, entailment, neutral]
        faith = float(probs[1] + 0.5 * probs[2])
        per_variant.append({
            "entail_prob":   round(float(probs[1]), 4),
            "neutral_prob":  round(float(probs[2]), 4),
            "contra_prob":   round(float(probs[0]), 4),
            "score":         round(max(0.0, min(1.0, faith)), 4),
        })

    mean = float(np.mean([d["score"] for d in per_variant])) if per_variant else 0.0
    return mean, per_variant


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
    """LLM-as-NLI: classify each response as entailment/neutral/contradiction.

    entailment→1.0, neutral→0.5, contradiction→0.0 (mirrors the deberta mapping).
    """
    from .llm_backend import get_chat_llm
    llm = get_chat_llm(_LLM_MODEL, quantization=os.getenv("JUDGE_QUANTIZATION"))
    msgs = [
        [
            {"role": "system", "content": "You check summary faithfulness. Given a "
             "SOURCE and a SUMMARY, reply with exactly one word: entailment (fully "
             "supported), neutral (not contradicted but not stated), or contradiction."},
            {"role": "user", "content": f"SOURCE:\n{premise}\n\nSUMMARY:\n{(r or '').strip()}\n\nLabel:"},
        ]
        for r in responses if (r or "").strip()
    ]
    if not msgs:
        return 0.0
    outs = llm.chat_batch(msgs, max_new_tokens=4)
    scores = []
    for o in outs:
        low = o.strip().lower()
        score = next((v for k, v in _LABEL_TO_SCORE.items() if k in low), 0.5)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0
