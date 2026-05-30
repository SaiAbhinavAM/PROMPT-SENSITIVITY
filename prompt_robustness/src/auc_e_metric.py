import numpy as np
from typing import List, Tuple, Dict, Union
from rouge_score import rouge_scorer as _rouge_mod

_ROUGE = _rouge_mod.RougeScorer(["rougeL"], use_stemmer=True)


def compute_auc_e_metric(
    reference: str,
    responses: List[str],
    return_curve: bool = False,
) -> Union[float, Tuple[float, Dict]]:
    """Performance Elasticity (AUC-E) — stability of ROUGE-L across variants.

    Definition (Flaw §3.4 clarification):
        AUC-E = 1 - CV(rougeL)
        where CV = std(rougeL_across_variants) / mean(rougeL_across_variants)

    Curve construction:
        x-axis: variant index (0..K-1) — discrete paraphrase order from
                the dataset (NOT a continuous perturbation strength).
        y-axis: ROUGE-L F-measure of response vs reference at that variant.
        AUC interpretation: 1 - CV summarizes the *flatness* of that curve;
                a flat (low-variance) curve → AUC-E close to 1.0.

    Args:
        reference: gold reference output (single string).
        responses: list of model responses, one per paraphrase variant.
        return_curve: if True, return (auc_e, {"variant_idx", "rougeL",
                      "mean", "std", "cv"}) so the metric is auditable.
                      Default False for backward compatibility.

    Returns:
        float in [0, 1] (or tuple when `return_curve=True`).
    """
    if not reference or not responses:
        if return_curve:
            return 0.0, {"variant_idx": [], "rougeL": [], "mean": 0.0, "std": 0.0, "cv": 0.0}
        return 0.0

    scores = []
    for resp in responses:
        if not resp.strip():
            scores.append(0.0)
            continue
        scores.append(_ROUGE.score(reference, resp)["rougeL"].fmeasure)

    if len(scores) < 2:
        val = float(scores[0]) if scores else 0.0
        if return_curve:
            return val, {
                "variant_idx": list(range(len(scores))),
                "rougeL": [float(s) for s in scores],
                "mean": val,
                "std": 0.0,
                "cv": 0.0,
            }
        return val

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    if mean < 1e-9:
        auc_e = 0.0
        cv = 0.0
    else:
        cv = std / mean
        # 1 - CV: CV=0 → perfect stability=1.0; CV=1 → complete instability=0.0
        auc_e = max(0.0, min(1.0, 1.0 - cv))

    if return_curve:
        return auc_e, {
            "variant_idx": list(range(len(scores))),
            "rougeL": [float(s) for s in scores],
            "mean": mean,
            "std": std,
            "cv": cv,
        }
    return auc_e
