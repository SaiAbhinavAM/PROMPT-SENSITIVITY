from typing import Dict


def compute_attribution_matrix(metrics: Dict[str, float]) -> str:
    """Combine ORI (Observable Robustness Index) and IFI (Intrinsic Fidelity Index)
    to classify the robustness of a sample.
    Simple rule‑based placeholder:
    - If all metrics >= 0.7 -> "True Robustness"
    - If any metric < 0.3 -> "Evaluation Artifact"
    - If PRI (computed elsewhere) is high but some metrics low -> "Stochastic Luck"
    - Otherwise -> "Knowledge Boundary"
    """
    # Determine basic categories
    low = any(v < 0.3 for v in metrics.values())
    high = all(v >= 0.7 for v in metrics.values())
    if high:
        return "True Robustness"
    if low:
        return "Evaluation Artifact"
    # Placeholder for stochastic luck detection (e.g., high PRI but mixed metrics)
    # Here we just check if average >= 0.6 but not all high
    avg = sum(metrics.values()) / len(metrics)
    if avg >= 0.6:
        return "Stochastic Luck"
    return "Knowledge Boundary"
