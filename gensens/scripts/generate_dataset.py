#!/usr/bin/env python3
"""
GenSens Dataset Generator — Main Entry Point
=============================================
Generates the full GenSens benchmark dataset by:
1. Loading source instances for each task
2. Generating paraphrase variants using LLaMA-3-8B-Instruct
3. Filtering with SBERT cosine similarity >= 0.82
4. Saving results as JSONL with periodic checkpointing

Usage:
    python scripts/generate_dataset.py --task all --n_instances 200 --n_variants 8
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Ensure scripts directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders import load_all_tasks, TASK_LOADERS
from paraphrase_generator import ParaphraseGenerator

# Canonical base instruction for summarization — used to seed the unique-instruction pool.
_CANONICAL_SUMM_INSTRUCTION = (
    "Summarize the following news article in 3-4 sentences, "
    "capturing the main events and key details."
)
_SUMM_UNIQUE_POOL_SIZE = 100  # max unique instructions to pre-generate

# Pre-generated pool file produced by generate_summ_instructions.py.
# When present, generate_dataset.py loads instructions from it instead of
# generating them on-the-fly with the paraphrase model.
_SUMM_POOL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "summ_instruction_pool.jsonl"
)


def setup_logging(task: str, log_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generation_{task}_{timestamp}.log")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(ch)

    logging.info(f"Logging to: {log_file}")
    return root_logger


def load_checkpoint(checkpoint_path: str) -> List[Dict[str, Any]]:
    """Load existing checkpoint if it exists."""
    if not os.path.exists(checkpoint_path):
        return []
    records = []
    with open(checkpoint_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_checkpoint(records: List[Dict[str, Any]], checkpoint_path: str):
    """Save records to checkpoint file."""
    with open(checkpoint_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_final(records: List[Dict[str, Any]], output_path: str):
    """Save final output as JSONL, plus a flat CSV alongside it.

    The CSV (one row per variant) lets the dataset be loaded for evaluation
    without re-running generation. Article/summary text is quoted by csv so
    commas/quotes/newlines round-trip losslessly.
    """
    import csv as _csv
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    csv_path = output_path.rsplit(".", 1)[0] + ".csv"
    cols = ["instance_id", "task", "variant_idx", "strategy", "base_text",
            "paraphrased_text", "sbert_similarity", "full_prompt",
            "input_text", "reference_output"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols, quoting=_csv.QUOTE_MINIMAL)
        w.writeheader()
        for rec in records:
            meta = rec.get("metadata", {})
            # Canonical bridge keys first; fall back to the legacy summarization
            # aliases so older datasets still export correctly.
            input_text = meta.get("input_text", meta.get("article", ""))
            reference_output = meta.get("reference_output", meta.get("gold_summary", ""))
            for v in rec.get("variants", []) or []:
                w.writerow({
                    "instance_id": rec.get("instance_id", ""),
                    "task": rec.get("task", ""),
                    "variant_idx": v.get("variant_idx", 0),
                    "strategy": v.get("strategy", ""),
                    "base_text": rec.get("base_text", ""),
                    "paraphrased_text": v.get("paraphrased_text", ""),
                    "sbert_similarity": v.get("sbert_similarity", 0.0),
                    "full_prompt": v.get("full_prompt", ""),
                    "input_text": input_text,
                    "reference_output": reference_output,
                })
    logging.getLogger(__name__).info(f"  Also wrote flat CSV: {csv_path}")


def compute_stats(all_records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute comprehensive statistics for all tasks."""
    stats = {}

    for task_name, records in all_records.items():
        if not records:
            stats[task_name] = {"n_instances": 0, "status": "no_records"}
            continue

        all_sims = []
        all_base_wc = []
        all_variant_wc = []
        total_variants = 0

        for rec in records:
            base_wc = len(rec["base_text"].split())
            all_base_wc.append(base_wc)
            for v in rec["variants"]:
                all_sims.append(v["sbert_similarity"])
                all_variant_wc.append(len(v["paraphrased_text"].split()))
                total_variants += 1

        sims = np.array(all_sims) if all_sims else np.array([0.0])
        base_wcs = np.array(all_base_wc)
        variant_wcs = np.array(all_variant_wc) if all_variant_wc else np.array([0.0])

        stats[task_name] = {
            "n_instances": len(records),
            "total_variants": total_variants,
            "avg_variants_per_instance": round(total_variants / len(records), 2),
            "sbert_similarity": {
                "mean": round(float(sims.mean()), 4),
                "std": round(float(sims.std()), 4),
                "min": round(float(sims.min()), 4),
                "max": round(float(sims.max()), 4),
                "p25": round(float(np.percentile(sims, 25)), 4),
                "p75": round(float(np.percentile(sims, 75)), 4),
            },
            "base_text_word_count": {
                "mean": round(float(base_wcs.mean()), 2),
                "std": round(float(base_wcs.std()), 2),
            },
            "variant_word_count": {
                "mean": round(float(variant_wcs.mean()), 2),
                "std": round(float(variant_wcs.std()), 2),
            },
        }

    return stats


def print_stats_table(stats: Dict[str, Any]):
    """Print a formatted statistics table."""
    print("\n" + "=" * 100)
    print("GENERATION STATISTICS")
    print("=" * 100)

    header = f"{'Task':<16} {'Inst':>6} {'Vars':>6} {'Avg/I':>6} " \
             f"{'SimMean':>8} {'SimStd':>8} {'SimMin':>8} {'SimMax':>8} " \
             f"{'BW_μ':>7} {'VW_μ':>7}"
    print(header)
    print("-" * 100)

    for task_name, s in stats.items():
        if task_name.startswith("_"):   # skip _meta and other internal keys
            continue
        if s.get("status") == "no_records":
            print(f"{task_name:<16} {'N/A':>6}")
            continue

        sim = s["sbert_similarity"]
        bw = s["base_text_word_count"]
        vw = s["variant_word_count"]

        row = (
            f"{task_name:<16} "
            f"{s['n_instances']:>6} "
            f"{s['total_variants']:>6} "
            f"{s['avg_variants_per_instance']:>6.1f} "
            f"{sim['mean']:>8.4f} "
            f"{sim['std']:>8.4f} "
            f"{sim['min']:>8.4f} "
            f"{sim['max']:>8.4f} "
            f"{bw['mean']:>7.1f} "
            f"{vw['mean']:>7.1f}"
        )
        print(row)

    print("=" * 100)


def _load_pool_file(path: str, n: int, logger: logging.Logger) -> Optional[List[str]]:
    """Load pre-generated instructions from summ_instruction_pool.jsonl if it exists.

    Instructions are sorted by complexity_level (1→4) so assignments go
    simple-to-complex across instances. Returns None if the file is absent.
    """
    if not os.path.exists(path):
        return None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None
    records.sort(key=lambda r: (r.get("complexity_level", 2), r.get("sbert_similarity", 0)))
    pool = [r["instruction"] for r in records]
    logger.info(
        f"  Loaded {len(pool)} instructions from pre-generated pool "
        f"({path}) — levels: "
        + ", ".join(
            f"{lvl}×{sum(1 for r in records if r.get('complexity_level') == lvl)}"
            for lvl in sorted({r.get('complexity_level', 0) for r in records})
        )
    )
    return pool


def _generate_unique_instructions(
    generator: ParaphraseGenerator,
    n: int,
    logger: logging.Logger,
) -> List[str]:
    """Return n unique summarization instructions for per-instance assignment.

    Priority order:
      1. Load from gensens/data/summ_instruction_pool.jsonl if produced by
         generate_summ_instructions.py (Qwen 8B, simple→complex).
      2. Fall back to ParaphraseGenerator.generate_instruction_pool (LLM
         freeform for llama/vllm, strategy-based for local/Flan-T5).
    Cycles if the pool is smaller than n.
    """
    pool = _load_pool_file(_SUMM_POOL_FILE, n, logger)

    if pool is None:
        logger.info("  Pre-generated pool not found; generating on-the-fly...")
        pool = generator.generate_instruction_pool(n, _CANONICAL_SUMM_INSTRUCTION, logger)

    if len(pool) < n:
        logger.warning(f"  Pool smaller than target ({len(pool)} < {n}); cycling to fill.")
        while len(pool) < n:
            pool += pool[: n - len(pool)]
    return pool[:n]


def run_task(
    task_name: str,
    generator: ParaphraseGenerator,
    n_instances: int,
    n_variants: int,
    seed: int,
    checkpoint_every: int,
    data_dir: str,
) -> List[Dict[str, Any]]:
    """Run paraphrase generation for a single task."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting task: {task_name}")
    logger.info(f"  n_instances={n_instances}, n_variants={n_variants}, seed={seed}")
    logger.info(f"{'='*60}")

    # Load source instances
    try:
        instances = TASK_LOADERS[task_name](n=n_instances, seed=seed)
    except Exception as e:
        logger.error(f"Failed to load data for task '{task_name}': {e}", exc_info=True)
        return []

    if not instances:
        logger.warning(f"No instances loaded for task '{task_name}'")
        return []

    logger.info(f"Loaded {len(instances)} source instances for '{task_name}'")

    # Summarization: assign one unique instruction paraphrase per instance so
    # cross-instance diversity is at instruction level, not just variant level.
    if task_name == "summarization":
        pool_size = min(len(instances), _SUMM_UNIQUE_POOL_SIZE)
        logger.info(f"  Generating unique instruction pool (pool_size={pool_size})...")
        unique_instructions = _generate_unique_instructions(generator, pool_size, logger)
        for i, instance in enumerate(instances):
            instr = unique_instructions[i % len(unique_instructions)]
            instance["base_text"] = instr
            instance["base_prompt"] = (
                instr
                + "\n\nArticle:\n"
                + instance["metadata"]["input_text"]
                + "\n\nSummary:"
            )

    # Check for existing checkpoint
    checkpoint_path = os.path.join(data_dir, f"gensens_{task_name}.checkpoint.jsonl")
    existing = load_checkpoint(checkpoint_path)
    completed_ids = {r["instance_id"] for r in existing}
    logger.info(f"Found {len(existing)} completed instances in checkpoint")

    results = list(existing)
    t_task_start = time.time()

    for i, record in enumerate(instances):
        if record["instance_id"] in completed_ids:
            logger.info(f"  Skipping {record['instance_id']} (already in checkpoint)")
            continue

        try:
            result = generator.process_instance(record, n_variants=n_variants)
            results.append(result)
            completed_ids.add(record["instance_id"])
        except Exception as e:
            logger.error(
                f"  Failed to process {record['instance_id']} after retries: {e}",
                exc_info=True,
            )
            # Save with empty variants
            results.append({
                "instance_id": record["instance_id"],
                "task": record["task"],
                "base_text": record["base_text"],
                "base_prompt": record["base_prompt"],
                "metadata": record["metadata"],
                "variants": [],
                "n_variants_generated": 0,
                "generation_time_s": 0.0,
            })

        # Periodic checkpoint
        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(results, checkpoint_path)
            logger.info(f"  ✓ Checkpoint saved at instance {i + 1}/{len(instances)}")

    task_time = time.time() - t_task_start
    logger.info(f"Task '{task_name}' completed in {task_time:.1f}s — {len(results)} records")

    # Save final output and remove checkpoint
    output_path = os.path.join(data_dir, f"gensens_{task_name}_{n_instances}inst_{n_variants}var.jsonl")
    save_final(results, output_path)
    logger.info(f"  Saved final output to: {output_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"  Removed checkpoint: {checkpoint_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GenSens Dataset Generator")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "summarization", "creative", "dialogue", "qa"],
        help="Task to generate (default: all)",
    )
    parser.add_argument("--n_instances", type=int, default=200, help="Instances per task")
    parser.add_argument("--n_variants", type=int, default=8, help="Variants per instance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Checkpoint frequency")
    parser.add_argument(
        "--model",
        type=str,
        default="vllm",
        choices=["llama", "vllm"],
        help=(
            "'vllm'  = batched vLLM generator (H100/A100, recommended); "
            "'llama' = HF transformers Llama 8B (single-seq, slower)"
        ),
    )
    parser.add_argument("--model-id", type=str, default=None,
                        help="HF repo id for the vLLM backend (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--quantization", type=str, default=None,
                        help="vLLM quantization, e.g. awq_marlin / gptq_marlin (None for bf16)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="vLLM fraction of GPU memory to use (default: 0.90)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="vLLM max model sequence length (default: 4096)")
    parser.add_argument("--similarity-threshold", type=float, default=0.82,
                        help="lower SBERT band edge")
    parser.add_argument("--similarity-upper", type=float, default=0.98,
                        help="upper SBERT band edge (reject near-identical)")
    parser.add_argument("--max-token-overlap", type=float, default=0.85,
                        help="reject candidates above this lexical (Jaccard) overlap")
    parser.add_argument("--max-per-strategy", type=int, default=2,
                        help="cap variants contributed by any single strategy "
                             "(legacy — superseded by --max-per-family from §6.6)")

    # ── vLLM multi-GPU / tensor parallelism ────────────────────────────────
    # PUBLICATION_SPEC §8 — required for 70B AWQ on A30 × 2 (single A30
    # has only 24GB, 70B AWQ needs ~35GB → must shard across 2 cards).
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor_parallel_size; set 2 for 70B AWQ on dual 24GB cards (A30 × 2)")

    # ── PUBLICATION_SPEC §6.1 / §7.1 — NLI gate + multi-NLI ensemble ───────
    parser.add_argument("--disable-nli-gate", dest="enable_nli_gate",
                        action="store_false", default=True,
                        help="DISABLE bidirectional NLI gate (legacy reproductions only)")
    parser.add_argument("--nli-entail-threshold", type=float, default=0.50,
                        help="Bidirectional entailment threshold τ_NLI (default 0.50)")
    parser.add_argument("--nli-model-name", type=str,
                        default="cross-encoder/nli-deberta-v3-small",
                        help="Primary NLI checkpoint (single-model mode)")
    parser.add_argument("--enable-nli-ensemble", action="store_true",
                        help="Enable 2-of-3 majority NLI ensemble (§7.1; "
                             "RECOMMENDED for publication-grade runs)")
    parser.add_argument("--nli-ensemble-models", nargs="+", default=None,
                        help="Override ensemble member checkpoints (default: "
                             "deberta-v3-small + deberta-v3-base + roberta-base)")
    parser.add_argument("--nli-ensemble-majority", type=int, default=2,
                        help="Majority threshold for ensemble vote (default 2 of 3)")

    # ── PUBLICATION_SPEC §6.2 — length-ratio filter ────────────────────────
    parser.add_argument("--length-ratio-min", type=float, default=0.70,
                        help="Minimum |candidate| / |base| word-count ratio")
    parser.add_argument("--length-ratio-max", type=float, default=1.50,
                        help="Maximum |candidate| / |base| word-count ratio")

    # ── PUBLICATION_SPEC §6.3 — SBERT semantic dedup ───────────────────────
    parser.add_argument("--semantic-dedup-threshold", type=float, default=0.95,
                        help="Reject candidate if SBERT cos > X to any accepted variant")

    # ── PUBLICATION_SPEC §6.5 — best-of-N per strategy ─────────────────────
    parser.add_argument("--best-of-n", type=int, default=3,
                        help="Number of vLLM candidates per (instance, strategy); "
                             "set 1 to recover pre-§6.5 single-shot behaviour")
    parser.add_argument("--best-of-n-temperature", type=float, default=0.95)
    parser.add_argument("--best-of-n-top-p", type=float, default=0.92)

    # ── PUBLICATION_SPEC §6.6 — per-family caps (4 families + back_translation) ─
    parser.add_argument("--max-per-family-lexical",   type=int, default=2)
    parser.add_argument("--max-per-family-syntactic", type=int, default=2)
    parser.add_argument("--max-per-family-pragmatic", type=int, default=1)
    parser.add_argument("--max-per-family-length",    type=int, default=1)

    # ── PUBLICATION_SPEC §6.3 — per-task SBERT threshold overrides ─────────
    # Use JSON: '{"dialogue": 0.78, "qa": 0.85}'. Defaults to the
    # built-in DEFAULT_TASK_THRESHOLDS (currently just dialogue: 0.78).
    parser.add_argument("--task-thresholds-json", type=str, default=None,
                        help='JSON mapping {task: lower SBERT threshold}; '
                             'e.g. \'{"dialogue": 0.78}\'')

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = str(project_root / "data")
    log_dir = str(project_root / "logs")
    os.makedirs(data_dir, exist_ok=True)

    # Setup logging
    setup_logging(args.task, log_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GenSens Dataset Generation Pipeline")
    logger.info(f"  Task:             {args.task}")
    logger.info(f"  Instances/task:   {args.n_instances}")
    logger.info(f"  Variants/inst:    {args.n_variants}")
    logger.info(f"  Seed:             {args.seed}")
    logger.info(f"  Checkpoint every: {args.checkpoint_every}")
    logger.info(f"  Model backend:    {args.model}")
    logger.info("=" * 60)

    # Determine tasks
    if args.task == "all":
        tasks = ["summarization", "creative", "dialogue", "qa"]
    else:
        tasks = [args.task]

    # Parse JSON task thresholds if provided.
    task_thresholds_override = None
    if args.task_thresholds_json:
        import json as _json
        try:
            task_thresholds_override = {
                str(k): float(v) for k, v in _json.loads(args.task_thresholds_json).items()
            }
            logger.info(f"  Task SBERT threshold overrides: {task_thresholds_override}")
        except Exception as e:
            logger.error(f"  invalid --task-thresholds-json: {e}; ignoring")
            task_thresholds_override = None

    # Compose per-family caps dict (PUBLICATION_SPEC §6.6). The
    # `back_translation` family is populated by a separate augmentation pass
    # (back_translation_family.py), so its cap is registered in
    # paraphrase_generator.MAX_PER_FAMILY but not exposed as a CLI knob here.
    max_per_family = {
        "lexical":   args.max_per_family_lexical,
        "syntactic": args.max_per_family_syntactic,
        "pragmatic": args.max_per_family_pragmatic,
        "length":    args.max_per_family_length,
    }

    # Load paraphrase generator ONCE (shared across all tasks)
    logger.info(f"Initializing ParaphraseGenerator (backend={args.model})...")
    logger.info(
        f"  Tier-2 ensemble: {'ON' if args.enable_nli_ensemble else 'OFF'} | "
        f"best_of_n={args.best_of_n} | "
        f"tensor_parallel_size={args.tensor_parallel_size} | "
        f"NLI gate={'ON' if args.enable_nli_gate else 'OFF'}"
    )
    t_init    = time.time()
    generator = ParaphraseGenerator(
        model_backend=args.model,
        similarity_threshold=args.similarity_threshold,
        similarity_upper=args.similarity_upper,
        max_token_overlap=args.max_token_overlap,
        max_per_strategy=args.max_per_strategy,
        model_id=args.model_id,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # Task 1 — multi-GPU sharding for 70B AWQ on A30 × 2.
        tensor_parallel_size=args.tensor_parallel_size,
        # Task 2 — PUBLICATION_SPEC Tier-1 / Tier-2 methodology knobs.
        enable_nli_gate=args.enable_nli_gate,
        nli_entail_threshold=args.nli_entail_threshold,
        nli_model_name=args.nli_model_name,
        enable_nli_ensemble=args.enable_nli_ensemble,
        nli_ensemble_models=args.nli_ensemble_models,
        nli_ensemble_majority=args.nli_ensemble_majority,
        length_ratio_min=args.length_ratio_min,
        length_ratio_max=args.length_ratio_max,
        semantic_dedup_threshold=args.semantic_dedup_threshold,
        best_of_n=args.best_of_n,
        best_of_n_temperature=args.best_of_n_temperature,
        best_of_n_top_p=args.best_of_n_top_p,
        max_per_family=max_per_family,
        task_thresholds=task_thresholds_override,
    )
    logger.info(f"ParaphraseGenerator initialized in {time.time() - t_init:.1f}s")

    # Run each task
    all_results = {}
    t_total_start = time.time()

    for task_name in tasks:
        task_results = run_task(
            task_name=task_name,
            generator=generator,
            n_instances=args.n_instances,
            n_variants=args.n_variants,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
            data_dir=data_dir,
        )
        all_results[task_name] = task_results

    total_time = time.time() - t_total_start

    # Compute and save stats
    stats = compute_stats(all_results)
    stats["_meta"] = {
        "total_time_s": round(total_time, 2),
        "n_instances_per_task": args.n_instances,
        "n_variants": args.n_variants,
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(),
    }

    stats_path = os.path.join(data_dir, "gensens_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to: {stats_path}")

    # Print stats table
    print_stats_table(stats)

    logger.info(f"\n✓ ALL DONE — Total pipeline time: {total_time:.1f}s")
    logger.info(f"  Output files in: {data_dir}/")

    # Summary counts
    total_instances = sum(len(r) for r in all_results.values())
    total_variants = sum(
        sum(rec["n_variants_generated"] for rec in recs)
        for recs in all_results.values()
    )
    logger.info(f"  Total instances: {total_instances}")
    logger.info(f"  Total variants:  {total_variants}")


if __name__ == "__main__":
    main()
