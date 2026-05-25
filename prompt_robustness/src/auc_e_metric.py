import numpy as np
from typing import List
from rouge_score import rouge_scorer as _rouge_mod

_ROUGE = _rouge_mod.RougeScorer(["rougeL"], use_stemmer=True)


def compute_auc_e_metric(reference: str, responses: List[str]) -> float:
    """Performance Elasticity (AUC-E) — variance of ROUGE-L across responses.

    Previous implementation was a proxy that collapsed to the mean ROUGE-L
    (Pearson r ≈ 0.996 with avg_rougeL), giving no additional signal.

    New definition: 1 - CV(rougeL), where CV = std/mean.
    - High score (→1): ROUGE-L is stable across prompt variants (robust).
    - Low score (→0): ROUGE-L varies wildly (sensitive to prompt wording).

    Returns a value in [0, 1]. Returns 0.0 if reference is empty.
    """
    if not reference or not responses:
        return 0.0

    scores = []
    for resp in responses:
        if not resp.strip():
            scores.append(0.0)
            continue
        scores.append(_ROUGE.score(reference, resp)["rougeL"].fmeasure)

    if len(scores) < 2:
        return float(scores[0]) if scores else 0.0

    mean = float(np.mean(scores))
    if mean < 1e-9:
        return 0.0

    cv = float(np.std(scores, ddof=1)) / mean
    # 1 - CV: CV=0 → perfect stability=1.0; CV=1 → complete instability=0.0
    return max(0.0, min(1.0, 1.0 - cv))
