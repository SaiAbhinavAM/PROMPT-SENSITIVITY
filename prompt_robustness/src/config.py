import os
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv

# Load env from the nearest .env walking up from the CWD (repo-root .env works
# whether run from prompt_robustness/ or elsewhere). Existing env vars win.
load_dotenv(find_dotenv(usecwd=True))

@dataclass
class Config:
    # Model configuration
    models: List[str] = field(default_factory=lambda: [m.strip() for m in os.getenv("MODEL_NAME", "google/flan-t5-base,google/flan-t5-large,sshleifer/distilbart-cnn-12-6,facebook/bart-large-cnn").split(",")])
    device: str = os.getenv("DEVICE", "cpu")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "50"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Generation parameters.
    # Deterministic (greedy) decoding is the default (rectification R2): output
    # variance across prompt variants must reflect PROMPT sensitivity, not
    # sampling noise. Set DO_SAMPLE=true only for explicit stochasticity studies.
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
    # Research-grade configuration flags (STEP 8, 10)
    # =========================================================================
    
    # Dynamic weighting: if True, final score uses adaptive PRI/Human weighting
    # based on USD (Utility-Stability Divergence). Default = False for backward compat.
    enable_dynamic_weighting: bool = False
    
    # Static weighting coefficients for Final_Score = w_pri * PRI + w_human * Human_Score
    final_score_pri_weight: float = 0.6
    final_score_human_weight: float = 0.4
    
    # Advanced metrics: enable research-grade metric variants
    enable_advanced_metrics: bool = True
    
    # Baseline metrics: ROUGE and BERTScore computation
    enable_rouge: bool = True
    enable_bertscore: bool = False  # Off by default (requires bert-score package)
    
    # Correlation analysis after benchmark
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

