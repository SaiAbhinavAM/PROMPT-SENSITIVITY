import os
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# GPU subject models (≥7B). Override via MODEL_NAME env var (comma-separated).
_GPU_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
]


def _resolve_default_models() -> List[str]:
    raw = os.getenv("MODEL_NAME", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(_GPU_MODELS)


@dataclass
class Config:
    # Model configuration
    models: List[str] = field(default_factory=_resolve_default_models)
    # Per-model quantization spec; models absent from this dict load in bf16/fp16.
    subject_quantizations: Dict[str, str] = field(default_factory=lambda: {
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": "awq_marlin",
    })
    device: str = os.getenv("DEVICE", "cuda")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "50"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Deterministic (greedy) decoding — output variance must reflect PROMPT
    # sensitivity, not sampling noise.
    do_sample: bool = os.getenv("DO_SAMPLE", "false").lower() == "true"
    seed: int = int(os.getenv("SEED", "42"))

    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(__file__))
    data_path: str = os.path.join(base_dir, "data", "sample_dataset.json")
    results_dir: str = os.path.join(base_dir, "results")
    cache_dir: str = os.path.join(base_dir, "cache")

    # Execution Flags
    enable_cache: bool = True
    enable_parallel: bool = True
    max_workers: int = 4

    # Metric weights (must sum to 1)
    weights: Dict[str, float] = None

    # =========================================================================
    # Research-grade configuration flags
    # =========================================================================

    enable_dynamic_weighting: bool = False

    final_score_pri_weight: float = 0.6
    final_score_human_weight: float = 0.4

    enable_advanced_metrics: bool = True
    enable_rouge: bool = True
    enable_bertscore: bool = False
    enable_correlation_analysis: bool = True

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "sms": 0.15,
                "auc_e": 0.15,
                "trd": 0.15,
                "kpig": 0.15,
                "ppl_var": 0.15,
                "bf": 0.15,
            }
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
