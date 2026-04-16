import re
from typing import List

def compute_hallucination_score(responses: List[str], input_text: str) -> float:
    """
    Hallucination Score (HS).
    Detects facts/words in the response that are entirely missing from the input text.
    Returns a score from 0.0 (no hallucinations) to 1.0 (highly hallucinated).
    """
    input_facts = set(re.findall(r'\b\w{4,}\b', input_text.lower()))
    if not input_facts: 
        return 0.0
        
    hs_scores = []
    for r in responses:
        resp_facts = set(re.findall(r'\b\w{4,}\b', r.lower()))
        if not resp_facts:
            hs_scores.append(0.0)
            continue
            
        hallucinated = len(resp_facts - input_facts)
        score = hallucinated / len(resp_facts)
        hs_scores.append(score)
        
    return sum(hs_scores) / len(hs_scores) if hs_scores else 0.0
