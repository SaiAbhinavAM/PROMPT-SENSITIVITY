import numpy as np
from typing import List
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_auc_score
from scipy.integrate import trapezoid as trapz
from sklearn.metrics.pairwise import cosine_similarity


def compute_auc_e_metric(reference: str, responses: List[str]) -> float:
    """Performance Elasticity (AUC‑E) metric.
    We define three perturbation levels (0.1, 0.2, 0.3) by artificially truncating
    the reference output and measuring similarity with each response.
    The similarity is cosine similarity between sentence embeddings (using a
    simple bag‑of‑words TF‑IDF vector for speed).
    The AUC of the (level, similarity) curve is returned.
    """
    # Simple vectorisation using character counts (placeholder for TF‑IDF)
    def vectorize(text: str) -> np.ndarray:
        chars = [ord(c) for c in text]
        return np.array(chars, dtype=float)

    ref_vec = vectorize(reference)
    sims = []
    levels = [0.1, 0.2, 0.3]
    for lvl in levels:
        # Truncate reference proportionally
        trunc_len = int(len(reference) * (1 - lvl))
        perturbed = reference[:max(1, trunc_len)]
        pert_vec = vectorize(perturbed)
        # Compute cosine similarity with each response and average
        level_sims = []
        for resp in responses:
            resp_vec = vectorize(resp)
            # pad to same length
            min_len = min(len(pert_vec), len(resp_vec))
            if min_len == 0:
                cos = 0.0
            else:
                cos = np.dot(pert_vec[:min_len], resp_vec[:min_len]) / (
                    np.linalg.norm(pert_vec[:min_len]) * np.linalg.norm(resp_vec[:min_len])
                )
            level_sims.append(cos)
        sims.append(np.mean(level_sims))
    # Compute AUC using trapezoidal rule
    auc = trapz(sims, levels) / (levels[-1] - levels[0])
    # Normalise to [0,1] (higher is better)
    return max(0.0, min(1.0, auc))
