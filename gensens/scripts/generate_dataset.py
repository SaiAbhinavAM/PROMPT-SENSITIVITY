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
from typing import List, Dict, Any

import numpy as np

# Ensure scripts directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders import load_all_tasks, TASK_LOADERS
from paraphrase_generator import ParaphraseGenerator


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
    """Save final output as JSONL."""
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


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
        choices=["all", "summarization", "code", "creative", "dialogue"],
        help="Task to generate (default: all)",
    )
    parser.add_argument("--n_instances", type=int, default=200, help="Instances per task")
    parser.add_argument("--n_variants", type=int, default=8, help="Variants per instance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Checkpoint frequency")
    parser.add_argument(
        "--model",
        type=str,
        default="local",
        choices=["local", "llama"],
        help=(
            "'local' = google/flan-t5-large (CPU/MPS, development on Mac); "
            "'llama' = Meta-Llama-3.1-8B-Instruct (H100, production)"
        ),
    )
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
        tasks = ["summarization", "code", "creative", "dialogue"]
    else:
        tasks = [args.task]

    # Load paraphrase generator ONCE (shared across all tasks)
    logger.info(f"Initializing ParaphraseGenerator (backend={args.model})...")
    t_init    = time.time()
    generator = ParaphraseGenerator(model_backend=args.model)
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
