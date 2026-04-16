import numpy as np
from typing import List

def compute_trd_metric(responses: List[str]) -> float:
    """
    Thematic Robustness Drift (TRD).
    Defined as the variance of token distributions/lengths across outputs.
    Returns normalized variance (0 to 1). Higher variance means higher drift.
    """
    if not responses or len(responses) < 2: 
        return 0.0
        
    lengths = [len(r.split()) for r in responses]
    avg_len = np.mean(lengths)
    
    if avg_len == 0: 
        return 0.0
        
    variance = np.var(lengths) / (avg_len ** 2) 
    return min(1.0, float(variance))
