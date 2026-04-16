from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingHelper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings (numpy array)."""
        return np.array(self.model.encode(texts, convert_to_numpy=True))
