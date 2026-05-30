import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Rebalanced CS composition (Flaw §2.4 — break the CS-coverage collinearity).
# Previously: CS = 0.5 * semantic + 0.5 * coverage  → r(CS, coverage) = 0.96.
# Reviewers can't read CS as a multi-axis quality signal when one axis
# dominates 96% of the variance. The new mix forces semantic similarity and
# length adequacy to each carry ≥15% of the variance:
CS_W_SEMANTIC = 0.40   # cosine(response, reference) — fluency-aware quality
CS_W_LENGTH   = 0.20   # length-adequacy vs reference (penalizes both too
                       # short and too long)
CS_W_COVERAGE = 0.40   # named-entity overlap vs reference — kept but no
                       # longer dominant
# Length-adequacy band: full credit when response length is between half and
# double the reference, linearly degraded outside.
_LEN_LOWER_RATIO = 0.5
_LEN_UPPER_RATIO = 2.0


def _length_adequacy(resp_len: int, ref_len: int) -> float:
    """Symmetric length-adequacy score in [0, 1].

    1.0 inside the band [0.5 * ref_len, 2 * ref_len]; linearly decays to 0
    at 0 length or at 4x ref_len. Reference length of 0 (rare/edge) returns
    a length-adequacy of 1.0 to avoid penalising legitimate empty references.
    """
    if ref_len <= 0:
        return 1.0
    if resp_len <= 0:
        return 0.0
    ratio = resp_len / ref_len
    if _LEN_LOWER_RATIO <= ratio <= _LEN_UPPER_RATIO:
        return 1.0
    if ratio < _LEN_LOWER_RATIO:
        # Linear from 0.0 (resp_len=0) to 1.0 (ratio = LOWER).
        return max(0.0, ratio / _LEN_LOWER_RATIO)
    # ratio > UPPER: linear from 1.0 (UPPER) → 0.0 (2 * UPPER = 4x ref_len)
    over = ratio - _LEN_UPPER_RATIO
    return max(0.0, 1.0 - over / _LEN_UPPER_RATIO)


def compute_correctness(responses: List[str], reference: str, embeddings: np.ndarray, embedder) -> Dict:
    """
    Computes a Correctness Score (CS) combining semantic similarity, length
    adequacy, and named-entity coverage (Flaw §2.4 rebalance).

        CS = 0.40 * semantic + 0.20 * length_adequacy + 0.40 * coverage
    """
    if not reference.strip() or len(responses) == 0:
        return {"cs_score": 0.0, "flags": [["missing_reference"]] * len(responses), "individual_scores": [0.0]*len(responses), "avg_length": 0.0, "avg_coverage": 0.0}

    ref_emb = embedder.encode([reference])[0]
    sims = cosine_similarity(embeddings, [ref_emb]).flatten()

    ref_tokens = len(reference.split())
    ref_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', reference))

    scores = []
    flags = []
    lengths = []
    coverages = []

    for i, resp in enumerate(responses):
        resp_flags = []
        semantic = max(0.0, float(sims[i]))

        # Coverage: named-entity overlap with reference.
        if ref_entities:
            resp_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', resp))
            coverage = len(ref_entities.intersection(resp_entities)) / len(ref_entities)
        else:
            coverage = 1.0
        coverages.append(coverage)

        # Length adequacy (new third axis — see _length_adequacy docstring).
        output_tokens = resp.split()
        resp_len = len(output_tokens)
        lengths.append(resp_len)
        length_score = _length_adequacy(resp_len, ref_tokens)

        cs_val = (
            CS_W_SEMANTIC * semantic
            + CS_W_LENGTH * length_score
            + CS_W_COVERAGE * coverage
        )
        cs_val = max(0.0, min(1.0, cs_val))

        # Existing degenerate-coverage gate retained: < 20% coverage caps
        # CS even after rebalancing, so a hallucinated-but-fluent response
        # cannot game the semantic+length axes alone.
        if coverage < 0.2:
            cs_val *= 0.6

        if cs_val < 0.25:
            resp_flags.append("irrelevant")

        scores.append(cs_val)
        flags.append(resp_flags)

    avg_cs = sum(scores) / len(scores) if scores else 0.0
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    avg_cov = sum(coverages) / len(coverages) if coverages else 0.0

    return {"cs_score": avg_cs, "flags": flags, "individual_scores": scores, "avg_length": avg_len, "avg_coverage": avg_cov}
