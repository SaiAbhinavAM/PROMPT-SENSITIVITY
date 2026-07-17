import re
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def extract_constraints(prompt: str) -> dict:
    """
    Extract simple constraints from a prompt.
    Detects word limits, length constraints, and format constraints.
    """
    constraints = {}
    prompt_lower = prompt.lower()
    
    # 1. Word limit constraints (e.g., "in 50 words", "limit to 100 words", "under 20 words")
    word_limit_match = re.search(r'(?:in|under|maximum|max|limit to)\s+(\d+)\s+words?', prompt_lower)
    if word_limit_match:
        constraints["word_limit"] = int(word_limit_match.group(1))
        
    # 2. Length constraints ("short", "brief", "detailed")
    if re.search(r'\b(short|brief|concise)\b', prompt_lower):
        constraints["length"] = "short"
    elif re.search(r'\b(detailed|long|comprehensive)\b', prompt_lower):
        constraints["length"] = "long"
        
    # 3. Format constraints ("bullet points", "list")
    if re.search(r'\b(bullet points|bullets|list)\b', prompt_lower):
        constraints["format"] = "list"
        
    return constraints

def check_constraints(response: str, constraints: dict) -> float:
    """
    Check if the response satisfies the extracted constraints.
    Returns a score between 0.0 and 1.0.
    """
    if not constraints:
        return 1.0 # No constraints to violate
        
    scores = []
    words = response.split()
    word_count = len(words)
    
    if "word_limit" in constraints:
        limit = constraints["word_limit"]
        # Allow a 10% margin of error
        if word_count <= limit * 1.1:
            scores.append(1.0)
        else:
            # Soft penalty based on how much it exceeded (capped at 0)
            penalty = max(0.0, 1.0 - ((word_count - limit) / limit))
            scores.append(penalty)
            
    if "length" in constraints:
        if constraints["length"] == "short":
            scores.append(1.0 if word_count < 100 else 0.5)
        elif constraints["length"] == "long":
            scores.append(1.0 if word_count >= 50 else 0.5)
            
    if "format" in constraints:
        if constraints["format"] == "list":
            # Check for bullet points, dashes, or numbered lists
            has_list = bool(re.search(r'(^|\n)(\s*[-*•]\s|\s*\d+\.\s)', response))
            scores.append(1.0 if has_list else 0.0)
            
    if not scores:
        return 1.0
        
    # Average the scores for all detected constraints
    return sum(scores) / len(scores)

def compute_kpig_metric(prompts: List[str], responses: List[str], alpha: float = 0.7, embedder=None) -> float:
    """
    Key Point Information Gain (KPIG) - Constraint-Aware Semantic Version.

    NOTE: the canonical pipeline uses compute_kpig_advanced from metrics_advanced.py.
    This function is retained as a lightweight fallback.

    Args:
        prompts: List of input prompts.
        responses: List of generated responses.
        alpha: Weight for semantic consistency vs constraint compliance (default: 0.7).
        embedder: Shared EmbeddingHelper instance. Created lazily if None.

    Returns:
        float: KPIG score between 0.0 and 1.0.
    """
    if not responses or all(not r.strip() for r in responses):
        return 0.0

    # --- 1. Compute Semantic Consistency ---
    if embedder is None:
        from .embeddings import EmbeddingHelper
        embedder = EmbeddingHelper()
    embeddings = embedder.encode(responses)
    
    if len(responses) > 1:
        sim_matrix = cosine_similarity(embeddings)
        # Average upper triangular part of similarity matrix
        upper_tri_indices = np.triu_indices(len(responses), k=1)
        pairwise_sims = sim_matrix[upper_tri_indices]
        semantic_consistency = max(0.0, float(np.mean(pairwise_sims)))
    else:
        semantic_consistency = 1.0
        
    # --- 2. Extract Constraints ---
    all_constraints = [extract_constraints(p) for p in prompts]
    
    # If no constraints detected in any prompt, fallback to semantic consistency only
    if not any(all_constraints):
        return max(0.0, min(1.0, semantic_consistency))
        
    # --- 3. Compute Constraint Consistency ---
    compliance_scores = []
    for resp, const in zip(responses, all_constraints):
        if const:
            score = check_constraints(resp, const)
            compliance_scores.append(score)
            
    if compliance_scores:
        constraint_consistency = sum(compliance_scores) / len(compliance_scores)
    else:
        constraint_consistency = 1.0
        
    # --- 4. Final KPIG Score ---
    kpig = alpha * semantic_consistency + (1.0 - alpha) * constraint_consistency
    
    return max(0.0, min(1.0, kpig))

if __name__ == "__main__":
    # Example usage and small test case
    test_prompts = [
        "Summarize this in 20 words.",
        "Give a detailed summary in 50 words using bullet points.",
        "Just summarize the text."
    ]
    test_responses = [
        "This is a very short summary about the text.",
        "- This is detailed.\n- It has bullet points.\n- It is within the 50 word limit.",
        "The text discusses various aspects of the topic in detail."
    ]
    
    score = compute_kpig_metric(test_prompts, test_responses)
    print(f"Test Prompts: {test_prompts}")
    print(f"Test Responses: {test_responses}")
    print(f"KPIG Score: {score:.4f}")
