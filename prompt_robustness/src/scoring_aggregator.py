import numpy as np
from typing import Dict

class ScoringAggregator:
    """Advanced metric aggregation alternatives to the standard PRI harmonic mean.
    Validates various mathematical reductions (geometric, arithmetic, harmonic).
    """
    def __init__(self):
        pass

    def arithmetic_mean(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Weighted arithmetic mean (highly compensatory - weak metrics are offset by strong)."""
        total_weight = sum(weights.values())
        if total_weight == 0: return 0.0
        weighted_sum = sum(scores[k] * (weights[k] / total_weight) for k in weights)
        return max(0.0, min(1.0, weighted_sum))
        
    def geometric_mean(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Weighted geometric mean (penalizes low scores significantly)."""
        total_weight = sum(weights.values())
        if total_weight == 0: return 0.0
        vals = []
        wts = []
        for k in weights:
            # Epsilon bounding blocks negative infinite logarithm outcomes
            val = max(1e-6, scores[k])
            vals.append(val)
            wts.append(weights[k] / total_weight)
            
        gm = np.exp(np.sum(np.array(wts) * np.log(vals)))
        return max(0.0, min(1.0, float(gm)))
        
    def harmonic_mean(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Weighted harmonic mean (strictly penalizes ANY failure). Base PRI standard."""
        total_weight = sum(weights.values())
        if total_weight == 0: return 0.0
        denominator = 0.0
        for k in weights:
            val = max(1e-6, scores[k])
            denominator += (weights[k] / total_weight) / val
            
        if denominator == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 / denominator))

    def aggregate(self, scores: Dict[str, float], weights: Dict[str, float], method: str = "harmonic") -> float:
        """Route to respective statistical logic based on chosen methodological configuration."""
        if method == "arithmetic":
            return self.arithmetic_mean(scores, weights)
        elif method == "geometric":
            return self.geometric_mean(scores, weights)
        else:
            return self.harmonic_mean(scores, weights)
