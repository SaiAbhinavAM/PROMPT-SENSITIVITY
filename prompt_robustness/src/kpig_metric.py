import re
from typing import List

def compute_kpig_metric(prompts: List[str], responses: List[str]) -> float:
    """
    Key Point Information Gain (KPIG).
    Measures the preservation of key facts across different responses.
    This continuous version computes (# key facts preserved) / (total key facts).
    """
    if not responses:
        return 0.0
        
    fact_sets = [set(re.findall(r'\b\w{4,}\b', r.lower())) for r in responses]
    if not any(fact_sets):
        return 1.0 # all empty, entirely consistent
        
    union_facts = set().union(*fact_sets)
    if not union_facts:
        return 1.0
    
    preservation_scores = []
    for fs in fact_sets:
        # How many of the total facts discovered in this run did this specific response preserve?
        preservation_scores.append(len(fs) / len(union_facts))
        
    return sum(preservation_scores) / len(preservation_scores)
