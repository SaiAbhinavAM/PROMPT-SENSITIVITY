import os
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingHelper:
    """Sentence-embedding helper for SMS / CS / TRD.

    The model is configurable (env EMBEDDER_MODEL). Defaults to the small
    all-MiniLM-L6-v2 for laptop runs; on the H100 set EMBEDDER_MODEL to a 7B
    embedder (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct) per the strict no-<2B
    policy. trust_remote_code is enabled so 7B embedders load.

    NOTE: switching embedder rescales SMS/KPIG/TRD (different geometry/dim), so
    scores are NOT comparable across embedder choices — re-baseline after a swap.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings (numpy array)."""
        return np.array(self.model.encode(texts, convert_to_numpy=True))
