import os
import numpy as np
from sentence_transformers import SentenceTransformer


def _patch_qwen2_config():
    """transformers >=5.0 moved rope_theta into rope_scaling dict; patch it back as an attribute."""
    try:
        from transformers import Qwen2Config
        if not hasattr(Qwen2Config, "rope_theta"):
            @property
            def _rope_theta(self):
                return (getattr(self, "rope_scaling", None) or {}).get("rope_theta", 10000.0)
            Qwen2Config.rope_theta = _rope_theta
    except Exception:
        pass


_patch_qwen2_config()


def _patch_dynamic_cache():
    """Patch DynamicCache for transformers >=5.0 API changes (methods removed/renamed)."""
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                return cls()
            DynamicCache.from_legacy_cache = from_legacy_cache
        if not hasattr(DynamicCache, "get_usable_length"):
            def get_usable_length(self, new_seq_length=None, layer_idx=0):
                try:
                    return self.get_seq_length(layer_idx)
                except Exception:
                    return 0
            DynamicCache.get_usable_length = get_usable_length
        if not hasattr(DynamicCache, "to_legacy_cache"):
            def to_legacy_cache(self):
                # transformers 5.7: layers is List[DynamicLayer], each with .keys/.values
                layers = getattr(self, "layers", None)
                if layers:
                    return tuple(
                        (layer.keys, layer.values)
                        for layer in layers
                        if getattr(layer, "is_initialized", False)
                    )
                # fallback for older 5.x that kept key_cache/value_cache
                key_cache = getattr(self, "key_cache", [])
                value_cache = getattr(self, "value_cache", [])
                return tuple(zip(key_cache, value_cache))
            DynamicCache.to_legacy_cache = to_legacy_cache
    except Exception:
        pass


_patch_dynamic_cache()


def _is_gte_qwen2(model_name: str) -> bool:
    n = model_name.lower()
    return "gte-qwen2" in n or "gte_qwen2" in n


class _GteQwen2Embedder:
    """
    gte-Qwen2 via native transformers (trust_remote_code=False) + EOS-token pooling.

    Why not SentenceTransformer(trust_remote_code=True)?
      - The custom modeling_qwen.py in the HF cache is incompatible with transformers >=5.7:
        (a) inv_freq (persistent=False buffer) is corrupted during from_pretrained → RoPE
            produces all-zero cos/sin → Q/K norms collapse to 0 → cosine sim ≈ 0.01
        (b) Left-padded batch attention mask is inverted → further breaks similarity.
      - trust_remote_code=False falls back to transformers' built-in Qwen2Model which has
        neither bug.  We then implement EOS-token pooling ourselves (the correct pooling
        strategy for this model family).
    """

    _BATCH = 8

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        # right-padding so that EOS index = (sum of mask) - 1
        self._tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False, padding_side="right"
        )
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=False,
            device_map="auto", torch_dtype=torch.float16
        )
        self._model.eval()

    def encode(self, texts: list) -> np.ndarray:
        import torch
        parts = []
        for i in range(0, len(texts), self._BATCH):
            chunk = texts[i: i + self._BATCH]
            enc = self._tok(
                chunk, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
            # index of the last non-padding token (works for right-padded batches)
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            emb = out.last_hidden_state[torch.arange(len(chunk)), last_idx]
            # L2-normalise
            emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            parts.append(emb.cpu().float().numpy())
        return np.concatenate(parts, axis=0)


class EmbeddingHelper:
    """Sentence-embedding helper for SMS / CS / TRD.

    Default: BAAI/bge-large-en-v1.5 (1024-dim, MTEB ~64, BGE paper Xiao et al. 2023).
    Override via EMBEDDER_MODEL env var.

    Cross-embedder ablation set (Flaw §3.5 — verify rankings are not an
    artifact of embedder choice; Spearman ρ across embedders should be > 0.7):

        EMBEDDER_MODEL=BAAI/bge-large-en-v1.5                 # default
        EMBEDDER_MODEL=Alibaba-NLP/gte-Qwen2-7B-instruct       # 7B, MTEB ~72
        EMBEDDER_MODEL=intfloat/e5-mistral-7b-instruct         # 7B, different family
        EMBEDDER_MODEL=mixedbread-ai/mxbai-embed-large-v1      # 335M, similar to BGE-large

    NOTE: switching embedder rescales SMS/KPIG/TRD (different geometry/dim),
    so scores are NOT comparable across embedder choices — re-baseline after a swap.
    """

    _DEFAULT = "BAAI/bge-large-en-v1.5"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDER_MODEL", self._DEFAULT)

        if _is_gte_qwen2(self.model_name):
            self._backend = _GteQwen2Embedder(self.model_name)
        else:
            trust = "qwen" in self.model_name.lower() or "gte" in self.model_name.lower()
            self._backend = SentenceTransformer(self.model_name, trust_remote_code=trust)

    def encode(self, texts: list) -> np.ndarray:
        if isinstance(self._backend, _GteQwen2Embedder):
            return self._backend.encode(texts)
        return np.array(self._backend.encode(texts, convert_to_numpy=True))
