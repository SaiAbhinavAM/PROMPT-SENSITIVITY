import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import EmbeddingHelper

class MultiReferenceEvaluator:
    """Multi-reference evaluation using semantic similarity.
    Calculates the maximum cosine similarity between the response and multiple valid references.
    """
    def __init__(self):
        self.embedder = EmbeddingHelper()
        
    def evaluate(self, references: List[str], response: str) -> float:
        if not references:
            return 0.0
        
        # Encode references and response via sentence transformers embeddings
        all_texts = references + [response]
        embeddings = self.embedder.encode(all_texts)
        
        ref_embeddings = embeddings[:-1]
        resp_embedding = embeddings[-1].reshape(1, -1)
        
        # Compute cosine similarities between the response and every reference
        similarities = cosine_similarity(resp_embedding, ref_embeddings)[0]
        
        # Return maximum similarity bounded strictly to [0,1]
        max_sim = np.max(similarities)
        return max(0.0, min(1.0, float(max_sim)))
