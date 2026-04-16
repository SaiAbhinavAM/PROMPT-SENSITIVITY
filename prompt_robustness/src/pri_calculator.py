from typing import Dict


def normalize_metric(value: float) -> float:
    """Clamp and ensure metric is within [0, 1]."""
    return max(0.0, min(1.0, value))


def compute_pri(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute Prompt Robustness Index (PRI) as weighted harmonic mean.
    PRI = sum(w_i) / sum(w_i / S_i)
    where S_i are normalized metric scores.
    """
    # Normalize metrics
    norm_metrics = {k: normalize_metric(v) for k, v in metrics.items()}
    # Ensure weights sum to 1 (or normalize)
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Sum of weights must be > 0")
    norm_weights = {k: w / total_weight for k, w in weights.items()}
    numerator = sum(norm_weights.values())
    denominator = sum(norm_weights[k] / norm_metrics.get(k, 1e-6) for k in norm_weights)
    pri = numerator / denominator if denominator != 0 else 0.0
    return normalize_metric(pri)
