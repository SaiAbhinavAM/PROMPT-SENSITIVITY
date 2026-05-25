#!/usr/bin/env python3
"""
Baseline Experiment for LL-PIRC Pipeline (Phase 2).

Loads 100 articles from CNN/DailyMail test split, generates K=5 paraphrase
variants of the summarization instruction via back-translation, runs greedy
decoding for each variant, and computes ROUGE-L vs gold summary.

Records:
- var_baseline[n] = variance of ROUGE-L across 5 outputs for article n
- IFI metrics (PPL_var, PC_stab) as a near-zero cost add-on

Saves results to:
- results/baseline.json
- results/ifi_metrics.json

Reference: method_analysis_prompt_sensitivity.md, Section 5 (lines 740-771)

Usage:
    python experiment_baseline.py [--config config.yaml] [--dry-run]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

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
logger = logging.getLogger("experiment_baseline")


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
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_cnn_dailymail(config: dict) -> list:
    """Load CNN/DailyMail test articles."""
    from datasets import load_dataset

    dataset_name = config['experiment']['dataset_name']
    dataset_config = config['experiment']['dataset_config']
    split = config['experiment']['dataset_split']

    logger.info(f"Loading {dataset_name}/{dataset_config}, split='{split}'")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    articles = []
    for item in dataset:
        articles.append({
            'article': item['article'],
            'gold_summary': item['highlights'],
            'id': item.get('id', '')
        })

    logger.info(f"Loaded {len(articles)} articles")
    return articles


# ═══════════════════════════════════════════════════════════════════════════════
# Paraphrase Generation (Back-Translation)
# ═══════════════════════════════════════════════════════════════════════════════

class BackTranslationParaphraser:
    """
    Generates paraphrases via back-translation through multiple pivot
    languages (EN → pivot → EN) using Helsinki-NLP opus-mt models.

    Each pivot language produces a slightly different paraphrase due to
    structural and lexical differences in the intermediate language.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Args:
            config: Full config dict (reads paraphrase section).
            device: Device for translation models (typically "cpu" to
                save GPU memory for the main model).
        """
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device
        self.pivot_languages = config['paraphrase']['pivot_languages']
        self.models = {}
        self.tokenizers = {}

        logger.info(
            f"Loading back-translation models for "
            f"{len(self.pivot_languages)} pivot languages..."
        )

        for pivot in self.pivot_languages:
            fwd_name = pivot['model_forward']
            bwd_name = pivot['model_backward']

            if fwd_name not in self.models:
                logger.info(f"  Loading {fwd_name}")
                self.tokenizers[fwd_name] = MarianTokenizer.from_pretrained(fwd_name)
                self.models[fwd_name] = MarianMTModel.from_pretrained(fwd_name).to(device)
                self.models[fwd_name].eval()

            if bwd_name not in self.models:
                logger.info(f"  Loading {bwd_name}")
                self.tokenizers[bwd_name] = MarianTokenizer.from_pretrained(bwd_name)
                self.models[bwd_name] = MarianMTModel.from_pretrained(bwd_name).to(device)
                self.models[bwd_name].eval()

    def translate(self, text: str, model_name: str) -> str:
        """Translate text using a specific model."""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=512)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def generate_paraphrase(self, text: str, pivot_idx: int) -> str:
        """Generate a single paraphrase via a specific pivot language."""
        pivot = self.pivot_languages[pivot_idx]
        # Forward: EN → pivot
        pivoted = self.translate(text, pivot['model_forward'])
        # Backward: pivot → EN
        paraphrased = self.translate(pivoted, pivot['model_backward'])
        return paraphrased

    def generate_paraphrases(
        self,
        text: str,
        K: int,
        similarity_model=None,
        similarity_threshold: float = 0.85
    ) -> List[str]:
        """
        Generate K valid paraphrases with semantic similarity filtering.

        Args:
            text: Original instruction text.
            K: Number of paraphrases to generate.
            similarity_model: SentenceTransformer model for filtering.
            similarity_threshold: Minimum cosine similarity to accept.

        Returns:
            paraphrases: List of K paraphrase strings (including original
                as the first element).
        """
        paraphrases = [text]  # Original is always first
        num_pivots = len(self.pivot_languages)

        if similarity_model is not None:
            original_embedding = similarity_model.encode(
                text, convert_to_tensor=True
            )

        for i in range(num_pivots):
            if len(paraphrases) >= K:
                break

            try:
                para = self.generate_paraphrase(text, i)

                # Skip if identical to original
                if para.strip().lower() == text.strip().lower():
                    logger.debug(
                        f"Pivot {i}: identical to original, skipping"
                    )
                    continue

                # Similarity filter
                if similarity_model is not None:
                    para_embedding = similarity_model.encode(
                        para, convert_to_tensor=True
                    )
                    sim = torch.nn.functional.cosine_similarity(
                        original_embedding.unsqueeze(0),
                        para_embedding.unsqueeze(0)
                    ).item()

                    if sim < similarity_threshold:
                        logger.debug(
                            f"Pivot {i}: similarity {sim:.3f} < "
                            f"{similarity_threshold}, skipping"
                        )
                        continue

                    logger.debug(
                        f"Pivot {i}: accepted (similarity={sim:.3f})"
                    )

                paraphrases.append(para)

            except Exception as e:
                logger.warning(f"Pivot {i} failed: {e}")
                continue

        # If we don't have enough, add slight variations of accepted ones
        retry = 0
        while len(paraphrases) < K and retry < 5:
            retry += 1
            # Re-translate an existing paraphrase through a different pivot
            src_idx = retry % (len(paraphrases) - 1) + 1 if len(paraphrases) > 1 else 0
            pivot_idx = retry % num_pivots
            try:
                para = self.generate_paraphrase(
                    paraphrases[src_idx] if src_idx < len(paraphrases) else text,
                    pivot_idx
                )
                if para.strip().lower() != text.strip().lower():
                    if similarity_model is not None:
                        para_embedding = similarity_model.encode(
                            para, convert_to_tensor=True
                        )
                        sim = torch.nn.functional.cosine_similarity(
                            original_embedding.unsqueeze(0),
                            para_embedding.unsqueeze(0)
                        ).item()
                        if sim >= similarity_threshold:
                            paraphrases.append(para)
                    else:
                        paraphrases.append(para)
            except Exception:
                continue

        if len(paraphrases) < K:
            logger.warning(
                f"Only generated {len(paraphrases)}/{K} valid paraphrases. "
                f"Padding with duplicates of last valid paraphrase."
            )
            while len(paraphrases) < K:
                paraphrases.append(paraphrases[-1])

        return paraphrases[:K]


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_target_model(config: dict):
    """Load the target LLM for summarization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config['model']['name']
    dtype_str = config['model']['dtype']
    device_map = config['model']['device_map']

    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

    logger.info(f"Loading target model: {model_name} ({dtype_str})")

    # Load with HF token from environment if available
    token = os.environ.get('HF_TOKEN', None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        token=token
    )
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded: {model_name}, device_map={device_map}")
    return model, tokenizer


def load_similarity_model(config: dict):
    """Load sentence-transformers model for paraphrase filtering."""
    from sentence_transformers import SentenceTransformer

    model_name = config['paraphrase']['similarity_model']
    logger.info(f"Loading similarity model: {model_name}")
    return SentenceTransformer(model_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_prompt(instruction: str, article: str, config: dict) -> str:
    """Format a prompt using the template from config."""
    template = config['prompts']['template']
    return template.format(instruction=instruction, article=article[:2000])


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_baseline_output(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200
) -> str:
    """Generate a single output using greedy decoding."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # greedy
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline_experiment(config: dict, dry_run: bool = False):
    """
    Run the full baseline experiment.

    For each article:
    1. Generate K paraphrase variants of the summarization instruction
    2. Format prompts using the template
    3. Generate outputs for each variant (greedy)
    4. Compute ROUGE-L vs gold summary
    5. Record variance of ROUGE-L across K outputs
    6. Compute IFI metrics (PPL_var, PC_stab)
    """
    K = config['paraphrase']['K']
    num_articles = 2 if dry_run else config['experiment']['num_articles']
    results_dir = Path(config['experiment']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = config['experiment']['checkpoint_interval']
    max_new_tokens = config['model']['max_new_tokens']
    base_instruction = config['prompts']['base_instruction']
    sim_threshold = config['paraphrase']['similarity_threshold']
    ifi_temperature = config['ifi']['temperature']

    # ─── Load data ────────────────────────────────────────────────────────
    if dry_run:
        # Override split for dry run
        config_copy = dict(config)
        config_copy['experiment'] = dict(config['experiment'])
        config_copy['experiment']['dataset_split'] = 'test[:2]'
        articles = load_cnn_dailymail(config_copy)
    else:
        articles = load_cnn_dailymail(config)

    # ─── Load models ──────────────────────────────────────────────────────
    logger.info("Loading models...")
    model, tokenizer = load_target_model(config)
    sim_model = load_similarity_model(config)
    paraphraser = BackTranslationParaphraser(config, device="cpu")

    # ─── Import IFI metrics ───────────────────────────────────────────────
    from src.ifi_metrics import compute_ifi_metrics

    # ─── Check for existing checkpoint ────────────────────────────────────
    checkpoint_path = results_dir / "baseline_checkpoint.json"
    results = []
    ifi_results = []
    start_idx = 0

    if checkpoint_path.exists() and not dry_run:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        results = checkpoint.get('results', [])
        ifi_results = checkpoint.get('ifi_results', [])
        start_idx = len(results)
        logger.info(f"Resuming from checkpoint at article {start_idx}")

    # ─── Main loop ────────────────────────────────────────────────────────
    total_start = time.time()

    for n in tqdm(range(start_idx, num_articles), desc="Baseline experiment"):
        if n >= len(articles):
            logger.warning(f"Only {len(articles)} articles available, stopping at {n}")
            break

        article_data = articles[n]
        article = article_data['article']
        gold_summary = article_data['gold_summary']

        logger.info(f"\n{'='*60}")
        logger.info(f"Article {n+1}/{num_articles}")
        logger.info(f"{'='*60}")

        # Step 1: Generate paraphrase variants of the instruction
        logger.info("Generating paraphrase variants...")
        paraphrase_instructions = paraphraser.generate_paraphrases(
            base_instruction, K, sim_model, sim_threshold
        )

        # Step 2: Format full prompts
        prompts = [
            format_prompt(instr, article, config)
            for instr in paraphrase_instructions
        ]

        # Step 3: Generate outputs for each variant
        rouge_scores = []
        outputs = []
        for k, prompt in enumerate(prompts):
            logger.info(f"  Generating output for variant {k+1}/{K}...")
            output = generate_baseline_output(
                model, tokenizer, prompt, max_new_tokens
            )
            outputs.append(output)

            # Step 4: Compute ROUGE-L
            rouge_l = compute_rouge_l(output, gold_summary)
            rouge_scores.append(rouge_l)
            logger.info(f"  Variant {k+1}: ROUGE-L = {rouge_l:.4f}")

        # Step 5: Record variance
        rouge_var = float(np.var(rouge_scores))
        rouge_mean = float(np.mean(rouge_scores))

        article_result = {
            'article_idx': n,
            'article_id': article_data.get('id', ''),
            'paraphrase_instructions': paraphrase_instructions,
            'outputs': outputs,
            'rouge_scores': rouge_scores,
            'rouge_var': rouge_var,
            'rouge_mean': rouge_mean,
            'gold_summary': gold_summary,
        }
        results.append(article_result)

        logger.info(
            f"  ROUGE-L: mean={rouge_mean:.4f}, var={rouge_var:.6f}"
        )

        # Step 6: Compute IFI metrics
        logger.info("  Computing IFI metrics...")
        try:
            ifi = compute_ifi_metrics(
                model, tokenizer, prompts, ifi_temperature
            )
            ifi_result = {
                'article_idx': n,
                'ppl_var': ifi['ppl_var'],
                'ppl_mean': ifi['ppl_mean'],
                'ppl_values': ifi['ppl_values'],
                'pc_stab': ifi['pc_stab'],
                'mean_branching_factors': ifi['mean_branching_factors'],
            }
            ifi_results.append(ifi_result)
            logger.info(
                f"  IFI: PPL_var={ifi['ppl_var']:.4f}, "
                f"PC_stab={ifi['pc_stab']:.6f}"
            )
        except Exception as e:
            logger.error(f"  IFI computation failed: {e}")
            ifi_results.append({
                'article_idx': n,
                'error': str(e)
            })

        # ─── Checkpoint ──────────────────────────────────────────────────
        if (n + 1) % checkpoint_interval == 0:
            logger.info(f"Saving checkpoint at article {n+1}...")
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'results': results,
                    'ifi_results': ifi_results,
                    'last_article': n
                }, f, indent=2, default=str)

    # ─── Save final results ──────────────────────────────────────────────
    total_time = time.time() - total_start

    baseline_output = {
        'config': {
            'model': config['model']['name'],
            'K': K,
            'num_articles': len(results),
            'similarity_threshold': sim_threshold,
            'paraphrase_method': config['paraphrase']['method'],
        },
        'results': results,
        'summary': {
            'mean_rouge_var': float(np.mean([r['rouge_var'] for r in results])),
            'mean_rouge_mean': float(np.mean([r['rouge_mean'] for r in results])),
            'total_time_seconds': total_time,
        }
    }

    baseline_path = results_dir / "baseline.json"
    with open(baseline_path, 'w') as f:
        json.dump(baseline_output, f, indent=2, default=str)
    logger.info(f"Baseline results saved to {baseline_path}")

    # Save IFI metrics
    ifi_output = {
        'config': {
            'model': config['model']['name'],
            'K': K,
            'temperature': ifi_temperature,
        },
        'results': ifi_results,
        'summary': {
            'mean_ppl_var': float(np.mean([
                r['ppl_var'] for r in ifi_results if 'ppl_var' in r
            ])) if ifi_results else None,
            'mean_pc_stab': float(np.mean([
                r['pc_stab'] for r in ifi_results if 'pc_stab' in r
            ])) if ifi_results else None,
        }
    }

    ifi_path = results_dir / "ifi_metrics.json"
    with open(ifi_path, 'w') as f:
        json.dump(ifi_output, f, indent=2, default=str)
    logger.info(f"IFI metrics saved to {ifi_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ─── Print summary ────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Articles processed: {len(results)}")
    logger.info(f"Mean ROUGE-L: {baseline_output['summary']['mean_rouge_mean']:.4f}")
    logger.info(f"Mean ROUGE-L variance: {baseline_output['summary']['mean_rouge_var']:.6f}")
    logger.info(f"Total time: {total_time:.1f}s")

    return baseline_output


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline experiment for LL-PIRC pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 2 articles for testing"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_baseline_experiment(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
