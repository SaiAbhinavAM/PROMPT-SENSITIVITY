import os
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingHelper:
    """Sentence-embedding helper for SMS / CS / TRD.

    Model: Alibaba-NLP/gte-Qwen2-7B-instruct (7B, GPU).
    Override via EMBEDDER_MODEL env var.

    NOTE: switching embedder rescales SMS/KPIG/TRD (different geometry/dim),
    so scores are NOT comparable across embedder choices — re-baseline after a swap.
    """

    _DEFAULT = "Alibaba-NLP/gte-Qwen2-7B-instruct"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDER_MODEL", self._DEFAULT)
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True))
