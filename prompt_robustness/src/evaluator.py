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
from .ppl_variance import compute_ppl_variance
from .branching_factor import compute_branching_factor
from .pri_calculator import compute_pri
from .attribution_matrix import compute_attribution_matrix
from .cache_manager import CacheManager
from . import scores as _scores

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
        return max(0.0, min(1.0, trust_pri * pri + trust_human * human_score))

    return _scores.compute_final_static(
        pri, human_score, config.final_score_pri_weight, config.final_score_human_weight
    )


def evaluate_sample(sample: Dict, config: Config, model_interface: ModelInterface, cache_manager: CacheManager, embedder: "EmbeddingHelper" = None) -> Dict:
    """Run the full evaluation pipeline for a single sample.

    Generation vs scoring: if ``sample['precomputed_responses']`` is present
    (loaded from responses.csv), generation is SKIPPED and the sample is scored
    directly — so re-evaluation needs no generation model. In that mode
    ``model_interface`` may be None; the model-internal PPL/BF diagnostics are
    then skipped (they are IFI-only and never enter PRI).
    """
    input_text = sample["input_text"]
    reference = sample.get("reference_output", "")
    model_name = sample.get("model") or (model_interface.model_name if model_interface else "unknown")
    instance_id = sample.get("instance_id", "")
    topic_label = sample.get("topic_label", sample.get("task", ""))
    strategies = sample.get("strategies")

    # 1. Obtain prompt variants.
    # Rectification R1 (close the loop): if the sample carries real GenSens
    # paraphrase variants, evaluate against THOSE. Only fall back to the
    # synthetic d1/d2/d3 templates when no variants are supplied (legacy JSON).
    provided_variants = sample.get("prompt_variants")
    if provided_variants:
        prompts = list(provided_variants)
        print(f"Using {len(prompts)} GenSens paraphrase variants")
    else:
        prompt_groups = generate_prompt_variants(input_text, max_per_level=None)
        prompts = flatten_prompt_variants(prompt_groups)
        print(f"Total prompts generated: {len(prompts)} (synthetic d1/d2/d3)")

    # 2. Obtain model responses.
    precomputed = sample.get("precomputed_responses")
    if precomputed is not None:
        # Score-from-CSV mode: responses already generated on the H100.
        responses = list(precomputed)
    else:
        responses = []
        uncached_indices = []
        uncached_prompts = []
        for i, p in enumerate(prompts):
            cached_resp = cache_manager.get("resp", model=model_name, prompt=p) if cache_manager else None
            if cached_resp:
                responses.append(cached_resp)
            else:
                responses.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_prompts.append(p)
        if uncached_prompts:
            batch_responses = model_interface.generate_responses(uncached_prompts)
            for idx, p, r in zip(uncached_indices, uncached_prompts, batch_responses):
                if cache_manager:
                    cache_manager.set(r, "resp", model=model_name, prompt=p)
                responses[idx] = r

    # 3. Compute consistency / quality / drift metrics.
    embedder = embedder or EmbeddingHelper()
    embeddings = embedder.encode(responses)

    # R6: the canonical TRD is embedding-based semantic drift, not length variance.
    trd_semantic = compute_trd_semantic(responses, embedder=embedder)
    trd_length = compute_trd_metric(responses)  # retained as a diagnostic only

    # R3: the canonical KPIG is reference-coverage (non-collinear with SMS).
    # The old cosine KPIG duplicated SMS and is no longer a score component.
    kpig_coverage = compute_kpig_advanced(
        prompts, responses, reference=reference, embedder=embedder
    )

    # Flaw §3.4: AUC-E also returns its underlying curve points so the
    # composite is auditable. The scalar feeds the score; the curve goes
    # into the result dict for downstream reviewers / plotting.
    auc_e_scalar, auc_e_curve = compute_auc_e_metric(reference, responses, return_curve=True)

    metrics = {
        "sms": compute_sms_metric(embeddings),
        "auc_e": auc_e_scalar,
        "trd": trd_semantic,        # canonical TRD = semantic drift (R6)
        "kpig": kpig_coverage,      # canonical KPIG = reference coverage (R3)
        # PPL variance & branching factor are INTRA-MODEL diagnostics (R6):
        # they are model-internal and NOT comparable across architectures, so
        # they feed IFI only and never the cross-model PRI ranking. Skipped in
        # score-from-CSV mode (no generation model available).
        "ppl_var": compute_ppl_variance(prompts, responses, model_interface) if model_interface else 0.0,
        "bf": compute_branching_factor(responses, model_interface) if model_interface else 0.0,
    }

    advanced_metrics = {
        "trd_semantic": trd_semantic,
        "kpig_advanced": kpig_coverage,
    }
    # R3: SMS_Wasserstein removed — it was ~0.98 collinear with SMS.

    # 4. Quality (CS) vs reference — kept INDEPENDENT of consistency and
    # faithfulness (R3/R4): no KPIG multiplier, no hallucination multiplier.
    from .correctness_metric import compute_correctness

    correctness = compute_correctness(responses, reference, embeddings, embedder)
    cs_score = max(0.0, min(1.0, correctness["cs_score"]))
    flags = correctness["flags"]
    avg_len = correctness.get("avg_length", 0.0)
    avg_cov = correctness.get("avg_coverage", 0.0)

    # 5. Hallucination Score — DIAGNOSTIC ONLY now (R4). It is no longer
    # multiplied into CS nor used as a PRI override; faithfulness is the single
    # faithfulness signal in PRI.
    from .hallucination_metric import compute_hallucination_score
    hs_score = max(0.0, min(1.0, compute_hallucination_score(responses, input_text, reference=reference)))

    # 6. Faithfulness axis (R3) — NLI entailment of each response by the source.
    # Falls back to the legacy (1 - HS) bag-of-words signal if NLI is unavailable.
    from .faithfulness_metric import compute_faithfulness_with_raw
    faithfulness, faithfulness_raw_outputs = compute_faithfulness_with_raw(responses, input_text)
    if faithfulness is None:
        faithfulness = 1.0 - hs_score
        faithfulness_raw_outputs = []
    faithfulness = max(0.0, min(1.0, faithfulness))

    # 7. PRI from three non-collinear axes (R3):
    #   consistency (SMS) ⟂ quality (CS vs reference) ⟂ faithfulness (NLI)
    consistency_score = max(0.0, min(1.0, metrics["sms"]))
    correctness_score = cs_score

    # Canonical PRI formula (single source of truth in src/scores.py; mirrored
    # by reaggregate.py). Short-output gate applied inside compute_pri.
    # Flaw §3.6: PRI weights are now ablatable via Config.pri_w_* (env vars).
    pri = _scores.compute_pri(
        consistency_score,
        correctness_score,
        faithfulness,
        avg_len,
        w_consistency=getattr(config, "pri_w_consistency", _scores.PRI_W_CONSISTENCY),
        w_quality=getattr(config, "pri_w_quality", _scores.PRI_W_QUALITY),
        w_faithfulness=getattr(config, "pri_w_faithfulness", _scores.PRI_W_FAITHFULNESS),
    )

    # Calculate confidence score
    confidence = (consistency_score + correctness_score + faithfulness) / 3.0
    
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
    
    # 6. LLM-as-a-Judge (Human_Score approximation).
    # R5: judge ALL prompt variants and average — the quality signal must span
    # every variant, not just responses[0]. Uses flan-t5-large (see llm_judge).
    from .llm_judge import llm_judge_with_raw
    human_score, judge_raw_outputs = llm_judge_with_raw(input_text, responses)

    # STEP 5: Utility-Stability Divergence
    usd_val = compute_usd(pri, human_score)

    # STEP 8: Final score with optional dynamic weighting
    final_score = _compute_final_score(pri, human_score, config, usd_val)

    # IFI — INTRA-MODEL diagnostic only (R6); not for cross-model ranking.
    ifi_score = _scores.compute_ifi(metrics["ppl_var"], metrics["bf"])

    # ORI — observable robustness (canonical formula in src/scores.py).
    ori_score = _scores.compute_ori(metrics["sms"], metrics["auc_e"], metrics["trd"], metrics["kpig"])

    # Dual-pillar diagnostic score for the publication framing. The existing
    # PRI remains the fair ranking score; Diagnostic_PRI is a strict harmonic
    # synthesis that exposes output-vs-internal failure modes.
    # Phase-6: harmonic sub-weights are sourced from Config so reviewers can
    # ablate ORI (SMS-heavy default) and IFI (PPL_var-heavy default) without
    # editing the formula module.
    diagnostic_ori = _scores.compute_diagnostic_ori(
        metrics["sms"], metrics["auc_e"], metrics["trd"], metrics["kpig"],
        w_sms=getattr(config,   "diag_ori_w_sms",   _scores.DIAG_ORI_W_SMS),
        w_auc_e=getattr(config, "diag_ori_w_auc_e", _scores.DIAG_ORI_W_AUC_E),
        w_kpig=getattr(config,  "diag_ori_w_kpig",  _scores.DIAG_ORI_W_KPIG),
        w_trd=getattr(config,   "diag_ori_w_trd",   _scores.DIAG_ORI_W_TRD),
    )
    diagnostic_ifi = _scores.compute_diagnostic_ifi(
        metrics["ppl_var"], metrics["bf"],
        w_ppl_var=getattr(config, "diag_ifi_w_ppl_var", _scores.DIAG_IFI_W_PPL_VAR),
        w_bf=getattr(config,      "diag_ifi_w_bf",      _scores.DIAG_IFI_W_BF),
    )
    diagnostic_pri = _scores.compute_diagnostic_pri(diagnostic_ori, diagnostic_ifi)
    diagnosis = _scores.compute_dual_pillar_diagnosis(diagnostic_ori, diagnostic_ifi)

    # STEP 7: Baseline metrics (ROUGE, BERTScore)
    rouge_scores = {}
    bertscore_scores = {}

    if config.enable_rouge:
        rouge_scores = compute_rouge_scores(responses, reference)

    if config.enable_bertscore:
        bertscore_scores = compute_bertscore(responses, reference)

    print(f"[{label}] PRI: {pri:.3f} | ORI: {ori_score:.3f} | IFI: {ifi_score:.3f} | Diagnostic_PRI: {diagnostic_pri:.3f} | Consistency: {consistency_score:.3f} | Correctness (CS): {correctness_score:.3f} | Faithfulness: {faithfulness:.3f} | HS(diag): {hs_score:.3f} | Human Score: {human_score:.3f} | Final: {final_score:.3f}")
    if interpretability_logs:
        for ilog in interpretability_logs:
            print(f"      -> {ilog}")
    print(f"      Avg Len: {avg_len:.1f} | Avg Cov: {avg_cov:.3f} | Confidence: {confidence:.3f}")

    # Log advanced metrics if available
    if advanced_metrics:
        trd_s = advanced_metrics.get('trd_semantic', 0.0)
        kpig_a = advanced_metrics.get('kpig_advanced', 0.0)
        print(f"      [Advanced] TRD_sem: {trd_s:.3f} | KPIG_adv: {kpig_a:.3f} | Faithfulness: {faithfulness:.3f} | USD: {usd_val:.3f}")

    if rouge_scores:
        print(f"      [ROUGE] R1: {rouge_scores.get('rouge1', 0):.3f} | R2: {rouge_scores.get('rouge2', 0):.3f} | RL: {rouge_scores.get('rougeL', 0):.3f}")

    attribution = compute_attribution_matrix(metrics)

    # STEP 9: Updated output JSON structure
    result = {
        "input_text": input_text,
        "reference_output": reference,
        "model": model_name,
        "instance_id": instance_id,
        "topic_label": topic_label,
        "strategies": strategies,
        "prompts": prompts,
        "responses": responses,
        "metrics": metrics,
        # Core scores
        "pri": pri,
        "ori_score": ori_score,
        "ifi_score": ifi_score,
        "diagnostic_ori": diagnostic_ori,
        "diagnostic_ifi": diagnostic_ifi,
        "diagnostic_pri": diagnostic_pri,
        "diagnosis": diagnosis,
        "sms": metrics["sms"],
        "trd": metrics["trd"],
        "kpig": metrics["kpig"],
        "ppl_var": metrics["ppl_var"],
        "bf": metrics["bf"],
        "cs": cs_score,
        "hs_score": hs_score,
        "faithfulness": faithfulness,
        "faithfulness_raw_outputs": faithfulness_raw_outputs,   # per-variant NLI raw scores
        "judge_raw_outputs": judge_raw_outputs,                 # per-variant 70B judge text + score
        "trd_length": trd_length,
        "human_score": human_score,
        "final_score": final_score,
        # Advanced / de-collinearized metrics (R3)
        "trd_semantic": advanced_metrics.get("trd_semantic", 0.0),
        "kpig_advanced": advanced_metrics.get("kpig_advanced", 0.0),
        "usd": usd_val,
        # Flaw §3.4: AUC-E underlying curve for auditability.
        "auc_e_curve": auc_e_curve,
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
