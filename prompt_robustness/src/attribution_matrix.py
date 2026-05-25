from typing import Dict

# Metrics where LOWER values are BETTER (must be inverted before
# applying higher-is-better thresholds).
_LOWER_IS_BETTER = {"trd", "ppl_var", "bf"}


def compute_attribution_matrix(metrics: Dict[str, float]) -> str:
    """Classify the robustness type of a sample using ORI sub-metrics.

    Categories:
    - "True Robustness"    — all adjusted metrics ≥ 0.65
    - "Evaluation Artifact"— any adjusted metric < 0.2
    - "Stochastic Luck"    — avg ≥ 0.55 but high spread across metrics
    - "Knowledge Boundary" — otherwise (moderate / mixed)

    Lower-is-better metrics (trd, ppl_var, bf) are inverted (1 - v)
    before thresholding so all values share the same polarity.
    """
    # Flip lower-is-better metrics
    adjusted = {}
    for k, v in metrics.items():
        if k in _LOWER_IS_BETTER:
            adjusted[k] = max(0.0, min(1.0, 1.0 - v))
        else:
            adjusted[k] = max(0.0, min(1.0, v))

    vals = list(adjusted.values())
    if not vals:
        return "Knowledge Boundary"

    low = any(v < 0.2 for v in vals)
    high = all(v >= 0.65 for v in vals)
    avg = sum(vals) / len(vals)

    if high:
        return "True Robustness"
    if low:
        return "Evaluation Artifact"

    # Stochastic Luck: average looks okay but metrics are inconsistent
    spread = max(vals) - min(vals)
    if avg >= 0.55 and spread > 0.35:
        return "Stochastic Luck"

    return "Knowledge Boundary"

