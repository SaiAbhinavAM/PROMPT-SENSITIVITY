import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_sms_metric(embeddings: np.ndarray, alpha: float = 0.5) -> float:
    """Compute Semantic Manifold Stability (SMS).

    SMS = similarity - alpha * diversity_penalty

    Embeddings are L2-normalised before the diversity penalty is
    computed so the penalty scale is independent of embedding magnitude.
    When all embeddings are (near-)identical the penalty is skipped and
    raw cosine similarity is returned, correctly giving SMS ≈ 1.0.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0

    # --- pairwise cosine similarity (scale-invariant) ---
    sim_matrix = cosine_similarity(embeddings)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    similarity = float(np.mean(upper_tri))

    # --- L2-normalise before computing diversity penalty ---
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)          # avoid division by zero
    emb_normed = embeddings / norms

    centroid = np.mean(emb_normed, axis=0)
    diversity_penalty = float(
        np.mean(np.linalg.norm(emb_normed - centroid, axis=1) ** 2)
    )

    # If outputs are effectively identical, skip the penalty
    if diversity_penalty < 1e-6:
        return max(0.0, min(1.0, similarity))

    sms = similarity - alpha * diversity_penalty
    return max(0.0, min(1.0, sms))
