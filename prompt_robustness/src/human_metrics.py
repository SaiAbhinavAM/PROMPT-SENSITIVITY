import math
from typing import List
from scipy.stats import pearsonr, spearmanr

class HumanMetrics:
    """Calculates alignment between programmatic robustness scores and human annotations."""
    
    def __init__(self):
        pass
        
    def calculate_human_agreement(self, programmatic_scores: List[float], manual_scores: List[float]) -> dict:
        """Returns statistical correlation methodologies (Pearson and Spearman)."""
        if len(programmatic_scores) != len(manual_scores) or len(programmatic_scores) < 2:
            return {"pearson": 0.0, "spearman": 0.0}
            
        try:
            p_corr, _ = pearsonr(programmatic_scores, manual_scores)
            s_corr, _ = spearmanr(programmatic_scores, manual_scores)
            
            # Handle NaN distributions if lists are uniformly constant
            if math.isnan(p_corr): p_corr = 0.0
            if math.isnan(s_corr): s_corr = 0.0
                
            return {
                "pearson": max(-1.0, min(1.0, float(p_corr))),
                "spearman": max(-1.0, min(1.0, float(s_corr)))
            }
        except Exception:
            return {"pearson": 0.0, "spearman": 0.0}
