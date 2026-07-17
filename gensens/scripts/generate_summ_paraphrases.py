"""
generate_summ_paraphrases.py
============================
Generate 5 paraphrase variants for each of the 50 seed summarization prompts.

Input:  gensens/data/gensens_summ_50_seed_prompts.jsonl  (35 Pool A + 15 Pool B)
Output: gensens/data/gensens_summ_50seed_5para.jsonl
Stats:  gensens/data/gensens_summ_stats.json
Checkpoint (resume): gensens/data/gensens_summ.checkpoint.jsonl

Default generator: meta-llama/Meta-Llama-3.1-8B-Instruct via vLLM (bf16,
single card, no quantization/tensor-parallel needed — fits a single ~24GB
GPU). This is scoped to this script only; generate_dataset.py's other
tasks (creative/dialogue/qa) still default to the 70B AWQ generator per
config.yaml. Override with --model-id for the 70B AWQ generator instead.

Usage:
    python gensens/scripts/generate_summ_paraphrases.py [options]

Options:
    --backend    vllm|llama         (default: vllm)
    --model-id   HF model id        (default: meta-llama/Meta-Llama-3.1-8B-Instruct)
    --n-variants int                (default: 5)
    --seed-file  path               (default: gensens/data/gensens_summ_50_seed_prompts.jsonl)
    --output     path               (default: gensens/data/gensens_summ_50seed_5para.jsonl)
    --resume                        Resume from checkpoint (default: True)
    --no-resume                     Restart from scratch
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Allow running from repo root or from within gensens/
_HERE = Path(__file__).resolve().parent
_GENSENS_ROOT = _HERE.parent
sys.path.insert(0, str(_GENSENS_ROOT.parent))  # repo root

from gensens.scripts.paraphrase_generator import ParaphraseGenerator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = _GENSENS_ROOT / "data"
DEFAULT_SEED_FILE   = DATA_DIR / "gensens_summ_50_seed_prompts.jsonl"
DEFAULT_OUTPUT      = DATA_DIR / "gensens_summ_50seed_5para.jsonl"
DEFAULT_CHECKPOINT  = DATA_DIR / "gensens_summ.checkpoint.jsonl"
DEFAULT_STATS       = DATA_DIR / "gensens_summ_stats.json"

# 8B generator, bf16, single-card — no AWQ quantization or tensor-parallel
# sharding needed (unlike the 70B AWQ generator used for other GenSens tasks).
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

CHECKPOINT_EVERY = 10   # write checkpoint after every N seeds
PROGRESS_EVERY   = 5    # print progress table row every N seeds

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_seeds(path: Path) -> List[Dict[str, Any]]:
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return dict keyed by prompt_id for already-completed seeds."""
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["prompt_id"]] = rec
    logger.info(f"Checkpoint loaded: {len(done)} seeds already done.")
    return done


def append_checkpoint(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_output_record(
    seed: Dict[str, Any],
    variants: List[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Produce a single output record per seed."""
    # Attach full_prompt to each variant: replace base_text in base_prompt
    base_text   = seed["base_text"]
    base_prompt = seed["base_prompt"]
    for v in variants:
        pt = v["paraphrased_text"]
        if base_text in base_prompt:
            v["full_prompt"] = base_prompt.replace(base_text, pt, 1)
        else:
            v["full_prompt"] = pt + "\n\n" + base_prompt

    return {
        "prompt_id":            seed["prompt_id"],
        "pool":                 seed["pool"],
        "dimension":            seed["dimension"],
        "complexity":           seed["complexity"],
        "base_text":            base_text,
        "base_prompt":          base_prompt,
        "task":                 seed["task"],
        "dataset":              seed["dataset"],
        "variants":             variants,
        "n_variants_generated": len(variants),
        "generation_time_s":    round(elapsed, 2),
    }


def verify_seeds(seeds: List[Dict[str, Any]]) -> None:
    n_a = sum(1 for s in seeds if s["pool"] == "A")
    n_b = sum(1 for s in seeds if s["pool"] == "B")
    if len(seeds) != 50 or n_a != 35 or n_b != 15:
        raise ValueError(
            f"Expected 50 seeds (35A + 15B), got {len(seeds)} "
            f"({n_a}A + {n_b}B). Check the seed file."
        )
    logger.info(f"Seed file verified: {len(seeds)} seeds ({n_a} Pool A, {n_b} Pool B).")


def print_progress_row(
    idx: int,
    seed: Dict[str, Any],
    n_gen: int,
    n_target: int,
    mean_sim: float,
    elapsed: float,
) -> None:
    flag = "" if n_gen >= n_target else f"  ⚠ only {n_gen}/{n_target}"
    logger.info(
        f"  [{idx+1:>3}/50] {seed['prompt_id']:>4}  pool={seed['pool']}  "
        f"variants={n_gen}/{n_target}  mean_sim={mean_sim:.3f}  "
        f"time={elapsed:.1f}s{flag}"
    )


def compute_stats(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-pool and per-dimension statistics."""
    by_pool: Dict[str, List] = {"A": [], "B": []}
    by_dim:  Dict[str, List] = {}
    flagged: List[str] = []

    for rec in results:
        pool = rec["pool"]
        dim  = rec["dimension"]
        n    = rec["n_variants_generated"]
        sims = [v["sbert_similarity"] for v in rec["variants"]]
        mean_sim = float(np.mean(sims)) if sims else 0.0

        by_pool[pool].append({"n": n, "mean_sim": mean_sim})
        by_dim.setdefault(dim, []).append({"n": n, "mean_sim": mean_sim})

        if n < 5:  # target is 5 variants per seed
            flagged.append(rec["prompt_id"])

    def agg(entries):
        ns   = [e["n"] for e in entries]
        sims = [e["mean_sim"] for e in entries]
        return {
            "count":    len(entries),
            "mean_n":   round(float(np.mean(ns)), 2),
            "min_n":    int(np.min(ns)),
            "max_n":    int(np.max(ns)),
            "mean_sim": round(float(np.mean(sims)), 4),
            "min_sim":  round(float(np.min(sims)), 4),
            "max_sim":  round(float(np.max(sims)), 4),
        }

    return {
        "total_seeds":     len(results),
        "n_pool_a":        len(by_pool["A"]),
        "n_pool_b":        len(by_pool["B"]),
        "pool_a":          agg(by_pool["A"]) if by_pool["A"] else {},
        "pool_b":          agg(by_pool["B"]) if by_pool["B"] else {},
        "per_dimension":   {dim: agg(entries) for dim, entries in by_dim.items()},
        "flagged_seeds":   flagged,
        "n_flagged":       len(flagged),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    seed_path  = Path(args.seed_file)
    out_path   = Path(args.output)
    ckpt_path  = Path(args.checkpoint)
    stats_path = Path(args.stats)

    seeds = load_seeds(seed_path)
    verify_seeds(seeds)

    done = load_checkpoint(ckpt_path) if args.resume else {}

    logger.info(
        f"Backend: {args.backend}  n_variants: {args.n_variants}  "
        f"Seeds to process: {len(seeds) - len(done)}"
    )

    gen_kwargs: Dict[str, Any] = dict(
        model_backend=args.backend,
        model_id=args.model_id,
        similarity_threshold=0.82,
        enable_nli_gate=True,
    )

    logger.info("Initialising ParaphraseGenerator …")
    generator = ParaphraseGenerator(**gen_kwargs)
    logger.info("Generator ready.")

    results: List[Dict[str, Any]] = list(done.values())
    done_ids = set(done.keys())
    n_written_checkpoint = 0

    for idx, seed in enumerate(seeds):
        pid = seed["prompt_id"]
        if pid in done_ids:
            logger.info(f"  [{idx+1:>3}/50] {pid} — skipped (checkpoint).")
            continue

        t0 = time.time()
        record_for_gen = {
            "instance_id": pid,
            "task":        seed["task"],
            "base_text":   seed["base_text"],
            "base_prompt": seed["base_prompt"],
            "pool":        seed["pool"],
            "metadata":    {
                "prompt_id":  pid,
                "pool":       seed["pool"],
                "dimension":  seed["dimension"],
                "complexity": seed["complexity"],
                "dataset":    seed["dataset"],
            },
        }

        variants = generator.generate_paraphrases(
            record_for_gen,
            n_variants=args.n_variants,
            max_retries=3,
        )
        elapsed = time.time() - t0

        out_rec = build_output_record(seed, variants, elapsed)
        results.append(out_rec)
        done_ids.add(pid)

        mean_sim = float(np.mean([v["sbert_similarity"] for v in variants])) if variants else 0.0
        if (idx + 1) % PROGRESS_EVERY == 0 or idx == 0:
            print_progress_row(idx, seed, len(variants), args.n_variants, mean_sim, elapsed)

        # Checkpoint every N seeds
        n_written_checkpoint += 1
        append_checkpoint(ckpt_path, out_rec)
        if n_written_checkpoint % CHECKPOINT_EVERY == 0:
            logger.info(f"  Checkpoint written ({n_written_checkpoint} new seeds).")

    # Write final output (all 50 seeds in seed-file order)
    pid_order = {s["prompt_id"]: i for i, s in enumerate(seeds)}
    results.sort(key=lambda r: pid_order.get(r["prompt_id"], 999))

    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Output written: {out_path}  ({len(results)} records)")

    # Stats
    stats = compute_stats(results)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written: {stats_path}")

    # Summary table
    logger.info("\n─── Generation summary ──────────────────────────────")
    for pool in ("A", "B"):
        p = stats[f"pool_{pool.lower()}"]
        if p:
            logger.info(
                f"  Pool {pool}: {p['count']} seeds  "
                f"n_variants mean={p['mean_n']:.1f} (min={p['min_n']})  "
                f"sim mean={p['mean_sim']:.3f}"
            )
    if stats["flagged_seeds"]:
        logger.warning(
            f"  ⚠ {stats['n_flagged']} seeds with <{args.n_variants} variants: "
            + ", ".join(stats["flagged_seeds"])
        )
    else:
        logger.info(f"  All seeds reached {args.n_variants} variants.")
    logger.info("─────────────────────────────────────────────────────\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate summarization paraphrases for 50 seed prompts.")
    p.add_argument("--backend",    default="vllm",   choices=["vllm", "llama"])
    p.add_argument("--model-id",   default=DEFAULT_MODEL_ID,
                   help="HF model id for the vLLM backend "
                        f"(default: {DEFAULT_MODEL_ID})")
    p.add_argument("--n-variants", type=int, default=5)
    p.add_argument("--seed-file",  default=str(DEFAULT_SEED_FILE))
    p.add_argument("--output",     default=str(DEFAULT_OUTPUT))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--stats",      default=str(DEFAULT_STATS))
    p.add_argument("--resume",     action="store_true",  default=True)
    p.add_argument("--no-resume",  dest="resume", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
