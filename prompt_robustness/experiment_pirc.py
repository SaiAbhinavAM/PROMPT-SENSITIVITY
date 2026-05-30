#!/usr/bin/env python3
"""
PIRC Experiment for LL-PIRC Pipeline (Phase 3).

Loads the same 100 articles and their 5 paraphrase variants from the
baseline experiment (results/baseline.json), then runs the full LL-PIRC
pipeline for each article:
  1. Detect sensitive layer ℓ* via S(ℓ) curve
  2. Identify anchor tokens at ℓ*
  3. Compute consensus hidden state across K variants
  4. Generate PIRC-stabilized outputs for every paraphrase variant
  5. Compute real post-PIRC ROUGE-L variance across variants

Hyperparameters:
- Test-time per-article ROUGE selection is disabled. Use fixed alpha/tau from
  config, or set pirc_selection.dev_tune_articles > 0 to choose alpha/tau on a
  held-out dev prefix before evaluating the remaining articles.

Saves results to results/pirc.json

Reference: method_analysis_prompt_sensitivity.md, Section 5 (lines 740-771)

Usage:
    python experiment_pirc.py [--config config.yaml] [--dry-run]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import yaml
import torch
import numpy as np
from tqdm import tqdm

# ─── Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("experiment_pirc")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# Load Baseline Results
# ═══════════════════════════════════════════════════════════════════════════════

def load_baseline_results(results_dir: str) -> Optional[dict]:
    """Load baseline results from Phase 2."""
    baseline_path = Path(results_dir) / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            data = json.load(f)
        logger.info(
            f"Loaded baseline results: {len(data['results'])} articles"
        )
        return data
    else:
        logger.warning(
            f"No baseline results found at {baseline_path}. "
            f"Run experiment_baseline.py first."
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_target_model(config: dict):
    """Load the target LLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config['model']['name']
    dtype_str = config['model']['dtype']
    device_map = config['model']['device_map']
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

    logger.info(f"Loading target model: {model_name} ({dtype_str})")

    token = os.environ.get('HF_TOKEN', None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        token=token
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_prompt(instruction: str, article: str, config: dict) -> str:
    """Format a prompt using the template from config."""
    template = config['prompts']['template']
    return template.format(instruction=instruction, article=article[:2000])


def load_article_text(config: dict, article_idx: int) -> str:
    """Load the article text for one baseline article index."""
    from datasets import load_dataset

    dataset = load_dataset(
        config['experiment']['dataset_name'],
        config['experiment']['dataset_config'],
        split=f"test[{article_idx}:{article_idx+1}]"
    )
    return dataset[0]['article']


def build_prompts_from_baseline_result(br: Dict, config: dict) -> List[str]:
    """Reconstruct full prompts for the paraphrase instructions in baseline.json."""
    article_text = load_article_text(config, br['article_idx'])
    return [
        format_prompt(instr, article_text, config)
        for instr in br['paraphrase_instructions']
    ]


def score_outputs_rouge_l(outputs: List[str], gold_summary: str) -> List[float]:
    """Compute ROUGE-L for each output against the gold summary."""
    return [compute_rouge_l(output or "", gold_summary) for output in outputs]


def run_pirc_for_article(
    pirc_generator,
    prompts: List[str],
    gold_summary: str,
    tau: float,
    alpha: float,
) -> Dict:
    """Run PIRC for all variants and return scored per-variant outputs."""
    pirc_result = pirc_generator.run_pipeline_all_variants(
        paraphrases=prompts,
        tau_override=tau,
        alpha_override=alpha,
    )
    outputs = pirc_result.get('outputs', [])
    rouge_scores = score_outputs_rouge_l(outputs, gold_summary)
    rouge_mean = float(np.mean(rouge_scores)) if rouge_scores else None
    rouge_var = float(np.var(rouge_scores)) if rouge_scores else None

    return {
        'pirc_outputs': outputs,
        'pirc_rouge_scores': rouge_scores,
        'pirc_rouge_mean': rouge_mean,
        'pirc_rouge_var': rouge_var,
        'ell_star': pirc_result['ell_star'],
        'S_curve': pirc_result['S_curve'],
        'num_anchors': pirc_result['num_anchors'],
        'anchor_fraction': pirc_result['anchor_fraction'],
        'alpha_used': pirc_result['alpha_used'],
        'tau_used': pirc_result['tau_used'],
    }


def resolve_pirc_hyperparams(
    config: dict,
    baseline_results: List[Dict],
    pirc_generator,
    dry_run: bool = False,
) -> Tuple[float, float, int, List[Dict]]:
    """Resolve alpha/tau without using test-set ROUGE for per-article selection.

    By default, this returns fixed config values. If
    pirc_selection.dev_tune_articles > 0, the first N baseline articles are used
    as a dev prefix to choose one global alpha/tau pair; those dev articles are
    then excluded from the final evaluation set.
    """
    selection_cfg = config.get('pirc_selection', {}) or {}
    tau_candidates = selection_cfg.get('tau_values') or [config['anchor_tokens']['tau']]
    alpha_candidates = selection_cfg.get('alpha_values') or [config['pirc']['alpha']]
    dev_tune_articles = int(selection_cfg.get('dev_tune_articles', 0) or 0)

    if dry_run:
        dev_tune_articles = 0

    fixed_tau = float(tau_candidates[0])
    fixed_alpha = float(alpha_candidates[0])
    if dev_tune_articles <= 0:
        logger.info(
            f"PIRC hyperparameters fixed before test: "
            f"tau={fixed_tau}, alpha={fixed_alpha}"
        )
        return fixed_tau, fixed_alpha, 0, baseline_results

    dev_n = min(dev_tune_articles, max(0, len(baseline_results) - 1))
    if dev_n == 0:
        logger.warning("Dev tuning requested but no held-out articles are available.")
        return fixed_tau, fixed_alpha, 0, baseline_results

    dev_results = baseline_results[:dev_n]
    eval_results = baseline_results[dev_n:]
    logger.info(
        f"Dev tuning PIRC on {dev_n} article(s); "
        f"final evaluation uses {len(eval_results)} held-out article(s)."
    )

    best = None
    for tau in tau_candidates:
        for alpha in alpha_candidates:
            scores = []
            variances = []
            for br in dev_results:
                try:
                    prompts = build_prompts_from_baseline_result(br, config)
                    result = run_pirc_for_article(
                        pirc_generator,
                        prompts,
                        br['gold_summary'],
                        float(tau),
                        float(alpha),
                    )
                    if result['pirc_rouge_mean'] is not None:
                        scores.append(result['pirc_rouge_mean'])
                    if result['pirc_rouge_var'] is not None:
                        variances.append(result['pirc_rouge_var'])
                except Exception as e:
                    logger.error(
                        f"Dev tuning failed for tau={tau}, alpha={alpha}: {e}"
                    )

            if not scores:
                continue

            mean_score = float(np.mean(scores))
            mean_var = float(np.mean(variances)) if variances else 0.0
            # Prefer quality, lightly penalize residual variance.
            objective = mean_score - mean_var
            logger.info(
                f"Dev candidate tau={tau}, alpha={alpha}: "
                f"mean_ROUGE-L={mean_score:.4f}, "
                f"mean_var={mean_var:.6f}, objective={objective:.4f}"
            )
            if best is None or objective > best['objective']:
                best = {
                    'tau': float(tau),
                    'alpha': float(alpha),
                    'objective': objective,
                }

    if best is None:
        logger.warning("No dev-tuned PIRC candidate succeeded; using fixed config values.")
        return fixed_tau, fixed_alpha, dev_n, eval_results

    logger.info(
        f"Selected global PIRC hyperparameters on dev: "
        f"tau={best['tau']}, alpha={best['alpha']}"
    )
    return best['tau'], best['alpha'], dev_n, eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# PIRC Pipeline Setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_pirc_pipeline(model, tokenizer, config: dict):
    """Initialize all LL-PIRC components."""
    from src.logit_lens import LogitLensExtractor
    from src.sensitive_layer import SensitiveLayerDetector
    from src.anchor_tokens import AnchorTokenIdentifier
    from src.pirc import PIRCGenerator

    # Logit Lens
    logit_lens = LogitLensExtractor(
        model,
        cast_to_float32=config['logit_lens']['cast_to_float32']
    )

    # Sensitive Layer Detector
    layer_detector = SensitiveLayerDetector(
        logit_lens,
        scan_start_fraction=config['sensitive_layer']['scan_start_fraction'],
        scan_end_fraction=config['sensitive_layer']['scan_end_fraction'],
        method=config['sensitive_layer']['method'],
        zscore_threshold=config['sensitive_layer']['zscore_threshold'],
        # Flaw §5.3 — honor a CLI/config force_ell_star override for ℓ* ablation.
        force_ell_star=config['sensitive_layer'].get('force_ell_star'),
    )

    # Anchor Token Identifier
    anchor_identifier = AnchorTokenIdentifier(
        logit_lens,
        tau=config['anchor_tokens']['tau'],
        tau_var=config['anchor_tokens']['tau_var']
    )

    # PIRC Generator
    pirc_generator = PIRCGenerator(
        model=model,
        tokenizer=tokenizer,
        logit_lens=logit_lens,
        layer_detector=layer_detector,
        anchor_identifier=anchor_identifier,
        alpha=config['pirc']['alpha'],
        soft_clamp_alpha=config['pirc']['soft_clamp_alpha'],
        max_new_tokens=config['model']['max_new_tokens']
    )

    return pirc_generator


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_pirc_experiment(config: dict, dry_run: bool = False):
    """
    Run the full PIRC experiment.

    For each article:
    1. Load paraphrase prompts from baseline results
    2. Resolve one global alpha/tau pair without test-set oracle selection
    3. Run LL-PIRC once per article and generate K stabilized outputs
    4. Compute ROUGE-L mean and variance across the K stabilized outputs
    """
    results_dir = Path(config['experiment']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = config['experiment']['checkpoint_interval']

    # ─── Load baseline results ────────────────────────────────────────────
    baseline = load_baseline_results(str(results_dir))
    if baseline is None:
        logger.error(
            "Cannot run PIRC experiment without baseline results. "
            "Run experiment_baseline.py first."
        )
        sys.exit(1)

    baseline_results = baseline['results']
    if dry_run:
        baseline_results = baseline_results[:2]

    baseline_mean_rouge = baseline['summary']['mean_rouge_mean']
    baseline_mean_var = baseline['summary']['mean_rouge_var']

    logger.info(
        f"Baseline: mean_ROUGE-L={baseline_mean_rouge:.4f}, "
        f"mean_var={baseline_mean_var:.6f}"
    )

    # ─── Load model and setup PIRC ────────────────────────────────────────
    model, tokenizer = load_target_model(config)
    pirc_generator = setup_pirc_pipeline(model, tokenizer, config)

    selected_tau, selected_alpha, dev_tune_articles, eval_baseline_results = (
        resolve_pirc_hyperparams(config, baseline_results, pirc_generator, dry_run)
    )

    # ─── Check for existing checkpoint ────────────────────────────────────
    # Flaw §4.4 — JSONL streaming checkpoint, fsync'd per record.
    from src.utils import JsonlCheckpointWriter
    checkpoint_path = results_dir / "pirc_checkpoint.json"  # legacy
    pirc_jsonl = results_dir / "pirc.jsonl"                  # per-article stream
    results = []
    start_idx = 0

    if pirc_jsonl.exists() and not dry_run:
        results = JsonlCheckpointWriter.read_all(str(pirc_jsonl))
        start_idx = len(results)
        logger.info(f"Resuming from JSONL checkpoint at article {start_idx}")
    elif checkpoint_path.exists() and not dry_run:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        results = checkpoint.get('results', [])
        start_idx = len(results)
        logger.info(f"Resuming from legacy checkpoint at article {start_idx}")

    pirc_stream = JsonlCheckpointWriter(str(pirc_jsonl)) if not dry_run else None

    # ─── Main loop ────────────────────────────────────────────────────────
    total_start = time.time()
    num_articles = len(eval_baseline_results)

    for n in tqdm(range(start_idx, num_articles), desc="PIRC experiment"):
        br = eval_baseline_results[n]
        article_idx = br['article_idx']
        gold_summary = br['gold_summary']

        logger.info(f"\n{'='*60}")
        logger.info(f"Article {n+1}/{num_articles} (idx={article_idx})")
        logger.info(f"{'='*60}")

        prompts = build_prompts_from_baseline_result(br, config)

        # ─── Record result ────────────────────────────────────────────────
        try:
            pirc_metrics = run_pirc_for_article(
                pirc_generator,
                prompts,
                gold_summary,
                selected_tau,
                selected_alpha,
            )
            pirc_var = pirc_metrics['pirc_rouge_var']
            baseline_var = br['rouge_var']
            variance_reduction = (
                1.0 - pirc_var / baseline_var
                if baseline_var and baseline_var > 0 and pirc_var is not None
                else 0.0
            )

            logger.info(
                f"  PIRC ROUGE-L: mean={pirc_metrics['pirc_rouge_mean']:.4f}, "
                f"var={pirc_var:.6f}, "
                f"variance_reduction={variance_reduction:.2%}, "
                f"ell*={pirc_metrics['ell_star']}, "
                f"anchors={pirc_metrics['num_anchors']}"
            )

            article_result = {
                'article_idx': article_idx,
                'split': 'test',
                'pirc_output': pirc_metrics['pirc_outputs'][0] if pirc_metrics['pirc_outputs'] else None,
                'pirc_outputs': pirc_metrics['pirc_outputs'],
                'pirc_rouge_l': pirc_metrics['pirc_rouge_mean'],
                'pirc_rouge_mean': pirc_metrics['pirc_rouge_mean'],
                'pirc_rouge_var': pirc_metrics['pirc_rouge_var'],
                'pirc_rouge_scores': pirc_metrics['pirc_rouge_scores'],
                'baseline_rouge_mean': br['rouge_mean'],
                'baseline_rouge_var': br['rouge_var'],
                'variance_reduction': variance_reduction,
                'ell_star': pirc_metrics['ell_star'],
                'S_curve': pirc_metrics['S_curve'],
                'num_anchors': pirc_metrics['num_anchors'],
                'anchor_fraction': pirc_metrics['anchor_fraction'],
                'alpha_used': pirc_metrics['alpha_used'],
                'tau_used': pirc_metrics['tau_used'],
            }
        except Exception as e:
            logger.error(f"  All PIRC attempts failed for article {article_idx}")
            import traceback
            traceback.print_exc()
            article_result = {
                'article_idx': article_idx,
                'split': 'test',
                'pirc_output': None,
                'pirc_rouge_l': None,
                'pirc_rouge_mean': None,
                'pirc_rouge_var': None,
                'pirc_rouge_scores': [],
                'baseline_rouge_mean': br['rouge_mean'],
                'baseline_rouge_var': br['rouge_var'],
                'error': str(e),
            }

        results.append(article_result)
        if pirc_stream is not None:
            pirc_stream.write(article_result)

        # ─── Checkpoint ──────────────────────────────────────────────────
        if (n + 1) % checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at article {n+1}...")
            with open(checkpoint_path, 'w') as f:
                json.dump({'results': results, 'last_article': n},
                          f, indent=2, default=str)

    # ─── Save final results ──────────────────────────────────────────────
    total_time = time.time() - total_start

    valid_results = [
        r for r in results
        if r.get('pirc_rouge_mean') is not None
        and r.get('pirc_rouge_var') is not None
    ]

    pirc_output = {
        'config': {
            'model': config['model']['name'],
            'K': config['paraphrase']['K'],
            'num_articles': len(results),
            'selected_tau': selected_tau,
            'selected_alpha': selected_alpha,
            'dev_tune_articles': dev_tune_articles,
            'selection_policy': (
                'dev_tuned_global' if dev_tune_articles > 0 else 'fixed_config'
            ),
        },
        'results': results,
        'summary': {
            'mean_pirc_rouge_l': float(np.mean([
                r['pirc_rouge_mean'] for r in valid_results
            ])) if valid_results else None,
            'mean_pirc_rouge_var': float(np.mean([
                r['pirc_rouge_var'] for r in valid_results
            ])) if valid_results else None,
            'mean_baseline_rouge_var': float(np.mean([
                r['baseline_rouge_var'] for r in valid_results
            ])) if valid_results else None,
            'mean_baseline_rouge_mean': float(np.mean([
                r['baseline_rouge_mean'] for r in valid_results
            ])) if valid_results else None,
            'num_successful': len(valid_results),
            'num_failed': len(results) - len(valid_results),
            'total_time_seconds': total_time,
            'mean_variance_reduction': float(np.mean([
                r['variance_reduction'] for r in valid_results
                if 'variance_reduction' in r
            ])) if valid_results else None,
            'ell_star_values': [
                r['ell_star'] for r in valid_results if 'ell_star' in r
            ],
        }
    }

    pirc_path = results_dir / "pirc.json"
    with open(pirc_path, 'w') as f:
        json.dump(pirc_output, f, indent=2, default=str)
    logger.info(f"PIRC results saved to {pirc_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Flaw §4.4 — close streaming JSONL checkpoint (data already fsync'd).
    if pirc_stream is not None:
        pirc_stream.close()

    # ─── Print summary ────────────────────────────────────────────────────
    if valid_results:
        ell_stars = pirc_output['summary']['ell_star_values']
        logger.info(f"\n{'='*60}")
        logger.info("PIRC EXPERIMENT SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Articles processed: {len(results)}")
        logger.info(f"Successful: {len(valid_results)}")
        logger.info(
            f"Mean PIRC ROUGE-L: "
            f"{pirc_output['summary']['mean_pirc_rouge_l']:.4f}"
        )
        logger.info(
            f"Mean Baseline ROUGE-L: "
            f"{pirc_output['summary']['mean_baseline_rouge_mean']:.4f}"
        )
        logger.info(
            f"Mean PIRC ROUGE-L variance: "
            f"{pirc_output['summary']['mean_pirc_rouge_var']:.6f}"
        )
        logger.info(
            f"Mean variance reduction: "
            f"{pirc_output['summary']['mean_variance_reduction']:.2%}"
        )
        logger.info(
            f"ℓ* range: [{min(ell_stars)}, {max(ell_stars)}], "
            f"mean={np.mean(ell_stars):.1f}, "
            f"std={np.std(ell_stars):.1f}"
        )
        logger.info(f"Total time: {total_time:.1f}s")

    return pirc_output


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run PIRC experiment for LL-PIRC pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 2 articles for testing"
    )
    # Flaw §5.3 / §5.4 — explicit ablation knobs for ℓ* and α. These
    # override anything in config.yaml, so a single ablation matrix can be
    # driven from the CLI without forking the config.
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override PIRC clamping alpha (e.g. 0.1, 0.25, 0.5, 0.75, 1.0)"
    )
    parser.add_argument(
        "--ell-star", type=int, default=None,
        help="Force a specific ℓ* (e.g. 6, 9, 12, 18, 24, 28) bypassing detection"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.alpha is not None:
        config.setdefault('pirc', {})['alpha'] = float(args.alpha)
        logger.info(f"[ablation] PIRC alpha overridden via CLI: {args.alpha}")
    if args.ell_star is not None:
        config.setdefault('sensitive_layer', {})['force_ell_star'] = int(args.ell_star)
        logger.info(f"[ablation] ℓ* forced via CLI: {args.ell_star}")
    # Flaw §4.2 — deterministic seeding for reproducibility.
    try:
        from src.utils import set_global_seed
        set_global_seed(int(config.get("seed", 42)) if isinstance(config, dict) else 42)
    except Exception:
        pass
    run_pirc_experiment(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
