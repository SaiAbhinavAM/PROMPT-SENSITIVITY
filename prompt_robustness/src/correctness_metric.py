import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

def compute_correctness(responses: List[str], reference: str, embeddings: np.ndarray, embedder) -> Dict:
    """
    Computes a Correctness Score (CS) combining semantic similarity, factual overlap, and explicit named entity coverage.
    """
    if not reference.strip() or len(responses) == 0:
        return {"cs_score": 0.0, "flags": [["missing_reference"]] * len(responses), "individual_scores": [0.0]*len(responses), "avg_length": 0.0, "avg_coverage": 0.0}

    ref_emb = embedder.encode([reference])[0]
    sims = cosine_similarity(embeddings, [ref_emb]).flatten()
    
    ref_facts = set(re.findall(r'\b\w{4,}\b', reference.lower()))
    ref_tokens = len(reference.split())
    ref_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', reference))
    
    scores = []
    flags = []
    lengths = []
    coverages = []
    
    for i, resp in enumerate(responses):
        resp_flags = []
        semantic = max(0.0, float(sims[i]))
        
        # Coverage Metric
        if ref_entities:
            resp_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', resp))
            coverage = len(ref_entities.intersection(resp_entities)) / len(ref_entities)
        else:
            coverage = 1.0
        coverages.append(coverage)
        
        # Length Metric tracking
        output_tokens = resp.split()
        lengths.append(len(output_tokens))
        
        # CS = 0.5 * semantic_similarity + 0.5 * keyword_overlap
        keyword_overlap = coverage
        cs_val = (0.5 * semantic) + (0.5 * keyword_overlap)
        cs_val = max(0.0, min(1.0, cs_val))
        
        if coverage < 0.2:
            cs_val *= 0.6
            
        if cs_val < 0.25:
            resp_flags.append("irrelevant")
            
        scores.append(cs_val)
        flags.append(resp_flags)
        
    avg_cs = sum(scores) / len(scores) if scores else 0.0
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    avg_cov = sum(coverages) / len(coverages) if coverages else 0.0
    
    return {"cs_score": avg_cs, "flags": flags, "individual_scores": scores, "avg_length": avg_len, "avg_coverage": avg_cov}
