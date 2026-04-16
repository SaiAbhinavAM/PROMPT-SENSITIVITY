import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_sms_metric(embeddings: np.ndarray, alpha: float = 0.5) -> float:
    """Compute Semantic Manifold Stability (SMS).
    SMS = similarity - alpha * diversity_penalty
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0
        
    sim_matrix = cosine_similarity(embeddings)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    similarity = float(np.mean(upper_tri))
    
    # Variance of embeddings (diversity penalty)
    centroid = np.mean(embeddings, axis=0)
    diversity_penalty = float(np.mean(np.linalg.norm(embeddings - centroid, axis=1)**2))
    
    sms = similarity - alpha * diversity_penalty
    return max(0.0, min(1.0, sms))
