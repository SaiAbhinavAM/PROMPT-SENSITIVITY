"""
evaluator.py — Core evaluation pipeline for the Prompt Robustness Evaluation Framework.

Orchestrates per-sample evaluation by:
1. Generating prompt variants (d1, d2, d3 perturbation levels)
2. Collecting model responses (with caching)
3. Computing ORI metrics (SMS, AUC-E, TRD, KPIG, PPL variance, BF)
4. Computing advanced metrics (semantic TRD, Wasserstein SMS, advanced KPIG)
5. Correctness scoring with hallucination penalty
6. LLM-as-a-Judge human score approximation
7. Baseline metrics (ROUGE, BERTScore)
8. Final score computation with optional dynamic weighting
"""

import logging
import math
import warnings
from typing import List, Dict

try:
    from .config import Config
except ImportError:
    from config import Config

from .prompt_generator import generate_prompt_variants, flatten_prompt_variants
from .model_interface import ModelInterface
from .embeddings import EmbeddingHelper
from .sms_metric import compute_sms_metric
from .auc_e_metric import compute_auc_e_metric
from .trd_metric import compute_trd_metric
from .kpig_metric import compute_kpig_metric
from .ppl_variance import compute_ppl_variance
from .branching_factor import compute_branching_factor
from .pri_calculator import compute_pri
from .attribution_matrix import compute_attribution_matrix
from .cache_manager import CacheManager

# Advanced metrics (STEP 1, 2, 4, 5, 7)
from .metrics_advanced import (
    compute_trd_semantic,
    compute_sms_wasserstein,
    compute_kpig_advanced,
    compute_usd,
    compute_rouge_scores,
    compute_bertscore,
)

logger = logging.getLogger(__name__)


def _compute_final_score(pri: float, human_score: float, config: Config, usd_val: float = 0.0) -> float:
    """
    Compute Final_Score with optional dynamic weighting (STEP 8).

    Static mode (default):
        Final_Score = w_pri * PRI + w_human * Human_Score

    Dynamic mode (config.enable_dynamic_weighting = True):
        When USD is high (PRI and human disagree), we trust the human score more.
        When USD is low (they agree), we keep the default weighting.

        Adaptive formula:
            trust_human = base_human_weight + 0.3 * USD  (capped at 0.7)
            trust_pri = 1 - trust_human
            Final_Score = trust_pri * PRI + trust_human * Human_Score

    Args:
        pri: Prompt Robustness Index.
        human_score: LLM-as-a-Judge approximated human score.
        config: Config object with weighting parameters.
        usd_val: Utility-Stability Divergence value.

    Returns:
        float: Final score in [0, 1].
    """
    if config.enable_dynamic_weighting:
        # When PRI and human diverge (high USD), shift trust toward human judgment
        trust_human = min(0.7, config.final_score_human_weight + 0.3 * usd_val)
        trust_pri = 1.0 - trust_human
        final = trust_pri * pri + trust_human * human_score
    else:
        final = config.final_score_pri_weight * pri + config.final_score_human_weight * human_score

    return max(0.0, min(1.0, final))


def evaluate_sample(sample: Dict, config: Config, model_interface: ModelInterface, cache_manager: CacheManager) -> Dict:
    """Run the full evaluation pipeline for a single sample."""
    input_text = sample["input_text"]
    reference = sample.get("reference_output", "")
    model_name = model_interface.model_name

    # 1. Generate prompt variants
    # For now, we flatten the d1, d2, d3 hierarchy so it works seamlessly 
    # with the rest of the metrics until AUC-E's custom integration is run.
    prompt_groups = generate_prompt_variants(input_text, max_per_level=None)
    prompts = flatten_prompt_variants(prompt_groups)
    
    total_prompts = sum(len(p) for p in prompt_groups.values())
    print(f"Total prompts generated: {total_prompts}")

    # 2. Generate model responses (with caching)
    responses = []
    uncached_indices = []
    uncached_prompts = []
    
    for i, p in enumerate(prompts):
        cached_resp = cache_manager.get("resp", model=model_name, prompt=p)
        if cached_resp:
            responses.append(cached_resp)
        else:
            responses.append(None) # Placeholder
            uncached_indices.append(i)
            uncached_prompts.append(p)
            
    if uncached_prompts:
        batch_responses = model_interface.generate_responses(uncached_prompts)
        for idx, p, r in zip(uncached_indices, uncached_prompts, batch_responses):
            cache_manager.set(r, "resp", model=model_name, prompt=p)
            responses[idx] = r

    # 3. Compute core ORI metrics (preserved from original)
    # Embeddings for SMS and advanced metrics
    embedder = EmbeddingHelper()
    embeddings = embedder.encode(responses)
    
    metrics = {
        "sms": compute_sms_metric(embeddings),
        "auc_e": compute_auc_e_metric(reference, responses),
        "trd": compute_trd_metric(responses),
        "kpig": compute_kpig_metric(prompts, responses),
        "ppl_var": compute_ppl_variance(prompts, responses, model_interface),
        "bf": compute_branching_factor(responses, model_interface),
    }

    # 3b. Compute advanced metrics (STEPS 1, 2, 4)
    advanced_metrics = {}
    if config.enable_advanced_metrics:
        # STEP 1: Semantic TRD (embedding-based topic drift)
        advanced_metrics["trd_semantic"] = compute_trd_semantic(responses, embedder=embedder)

        # STEP 2: Wasserstein SMS (approximate Earth-Mover's Distance)
        advanced_metrics["sms_wasserstein"] = compute_sms_wasserstein(embeddings)

        # STEP 4: Advanced KPIG (semantic information gain with redundancy penalty)
        advanced_metrics["kpig_advanced"] = compute_kpig_advanced(
            prompts, responses, reference=reference, embedder=embedder
        )

    # 4. Correctness score & Normalizations
    from .correctness_metric import compute_correctness
    
    correctness = compute_correctness(responses, reference, embeddings, embedder)
    cs_score = correctness["cs_score"]
    flags = correctness["flags"]
    avg_len = correctness.get("avg_length", 0.0)
    avg_cov = correctness.get("avg_coverage", 0.0)
    
    # Strengthen KPIG influence
    kpig_score = metrics["kpig"]
    cs_score = cs_score * (0.5 + 0.5 * kpig_score)
    cs_score = max(0.0, min(1.0, cs_score))
    
    # 5. Hallucination Metric
    from .hallucination_metric import compute_hallucination_score
    hs_score = compute_hallucination_score(responses, input_text)
    hs_score = max(0.0, min(1.0, hs_score))
    
    # Fix hallucination leakage into CS
    cs_score = cs_score * math.exp(-1.5 * hs_score)
    cs_score = max(0.0, min(1.0, cs_score))
    
    # New Interpretable PRI definitions
    consistency_score = metrics["sms"]
    consistency_score = max(0.0, min(1.0, consistency_score))
        
    correctness_score = cs_score
    
    # Calculate penalized robustness
    pri = (0.40 * consistency_score) + (0.35 * correctness_score) + (0.25 * math.exp(-hs_score))
        
    # Add strict hallucination override to PRI
    if hs_score > 0.5:
        pri *= 0.6
        
    # Penalize too-short outputs
    if avg_len < 12:
        pri *= 0.85
        
    pri = max(0.0, min(1.0, pri))
    
    # Calculate confidence score
    confidence = (consistency_score + correctness_score + (1.0 - hs_score)) / 3.0
    
    # Final labels
    label = "Unreliable"
    if pri >= 0.70:
        label = "Robust"
    elif pri >= 0.50:
        label = "Moderate"
        
    # Interpretability logs arrays
    interpretability_logs = []
    if avg_cov < 0.2:
        interpretability_logs.append("Very low coverage → incomplete summary")
    
    if avg_len < 12:
        interpretability_logs.append("Summary too short → lacks detail")
        
    if hs_score > 0.4:
        interpretability_logs.append("High hallucination detected → reliability reduced")
        
    # Add Sanity Checks & Logging
    if consistency_score < 0.1:
        warnings.warn("Consistency is extremely low (<0.1) - marking as unstable.")
        for f in flags: f.append("unstable")
    if hs_score > 0.5:
        warnings.warn("Hallucination Score is extremely high (>0.5) - marking as hallucinated.")
        for f in flags: f.append("hallucinated")
    
    # Final Correctness-Aware iPRI
    iPRI = pri * cs_score
    iPRI = max(0.0, min(1.0, iPRI))
    
    # 6. LLM-as-a-Judge (Human_Score approximation)
    from .llm_judge import llm_judge, parse_judge_output
    # Use best summary (first variant) to save compute
    best_summary = responses[0]
    judge_output = llm_judge(input_text, best_summary)
    human_score = parse_judge_output(judge_output)

    # STEP 5: Utility-Stability Divergence
    usd_val = compute_usd(pri, human_score)

    # STEP 8: Final score with optional dynamic weighting
    final_score = _compute_final_score(pri, human_score, config, usd_val)

    # STEP 7: Baseline metrics (ROUGE, BERTScore)
    rouge_scores = {}
    bertscore_scores = {}

    if config.enable_rouge:
        rouge_scores = compute_rouge_scores(responses, reference)

    if config.enable_bertscore:
        bertscore_scores = compute_bertscore(responses, reference)

    print(f"[{label}] PRI: {pri:.3f} | Consistency: {consistency_score:.3f} | Correctness (CS): {correctness_score:.3f} | Hallucination (HS): {hs_score:.3f} | Human Score: {human_score:.3f} | Final: {final_score:.3f}")
    if interpretability_logs:
        for ilog in interpretability_logs:
            print(f"      -> {ilog}")
    print(f"      Avg Len: {avg_len:.1f} | Avg Cov: {avg_cov:.3f} | Confidence: {confidence:.3f}")

    # Log advanced metrics if available
    if advanced_metrics:
        trd_s = advanced_metrics.get('trd_semantic', 0.0)
        sms_w = advanced_metrics.get('sms_wasserstein', 0.0)
        kpig_a = advanced_metrics.get('kpig_advanced', 0.0)
        print(f"      [Advanced] TRD_sem: {trd_s:.3f} | SMS_W: {sms_w:.3f} | KPIG_adv: {kpig_a:.3f} | USD: {usd_val:.3f}")

    if rouge_scores:
        print(f"      [ROUGE] R1: {rouge_scores.get('rouge1', 0):.3f} | R2: {rouge_scores.get('rouge2', 0):.3f} | RL: {rouge_scores.get('rougeL', 0):.3f}")

    attribution = compute_attribution_matrix(metrics)

    # STEP 9: Updated output JSON structure
    result = {
        "input_text": input_text,
        "reference_output": reference,
        "model": model_name,
        "prompts": prompts,
        "responses": responses,
        "metrics": metrics,
        # Core scores
        "pri": pri,
        "sms": metrics["sms"],
        "trd": metrics["trd"],
        "kpig": metrics["kpig"],
        "ppl_var": metrics["ppl_var"],
        "bf": metrics["bf"],
        "cs": cs_score,
        "hs_score": hs_score,
        "human_score": human_score,
        "final_score": final_score,
        # Advanced metrics (STEP 1, 2, 4, 5)
        "sms_wasserstein": advanced_metrics.get("sms_wasserstein", 0.0),
        "trd_semantic": advanced_metrics.get("trd_semantic", 0.0),
        "kpig_advanced": advanced_metrics.get("kpig_advanced", 0.0),
        "usd": usd_val,
        # Baseline metrics (STEP 7)
        "rouge": rouge_scores,
        "bertscore": bertscore_scores,
        # Preserved original fields
        "iPRI": iPRI,
        "consistency": consistency_score,
        "flags": flags,
        "label": label,
        "confidence": confidence,
        "avg_length": avg_len,
        "avg_coverage": avg_cov,
        "interpretability_logs": interpretability_logs,
        "attribution": attribution,
    }

    return result
