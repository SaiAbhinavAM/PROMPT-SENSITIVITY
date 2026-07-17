"""
regenerate_flagged.py
=====================
Re-generate paraphrases for seeds flagged by validate_summ_paraphrases.py.

Reads the validation report, identifies flagged seeds, and re-runs generation
with a tighter prompt (one-shot example injected) for up to MAX_ATTEMPTS retries.
Overwrites flagged records in-place in the main output file.
Seeds that still fail after MAX_ATTEMPTS are kept as-is and marked
  constraint_leakage_unresolved: true

After regeneration, re-runs validation and prints a diff of before/after flag counts.

Usage:
    python gensens/scripts/regenerate_flagged.py [options]

Options:
    --validation    path   (default: gensens/data/gensens_summ_validation.json)
    --data          path   (default: gensens/data/gensens_summ_50seed_5para.jsonl)
    --seeds         path   (default: gensens/data/gensens_summ_50_seed_prompts.jsonl)
    --output        path   output file path (default: same as --data, overwrite in-place)
    --backend       vllm|llama     (default: vllm)
    --model-id      path           override HF model id
    --max-attempts  int            (default: 3)
    --n-variants    int            (default: 5)
    --skip-validation              Skip re-validation after regeneration
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_GENSENS_ROOT = _HERE.parent
sys.path.insert(0, str(_GENSENS_ROOT.parent))

DATA_DIR = _GENSENS_ROOT / "data"

DEFAULT_VALIDATION = DATA_DIR / "gensens_summ_validation.json"
DEFAULT_DATA       = DATA_DIR / "gensens_summ_50seed_5para.jsonl"
DEFAULT_SEEDS      = DATA_DIR / "gensens_summ_50_seed_prompts.jsonl"
MAX_ATTEMPTS       = 3

# Same 8B generator as generate_summ_paraphrases.py (2026-07-05) — keeps
# regeneration consistent with the original generation run instead of
# silently falling back to the 70B AWQ generator.
DEFAULT_MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ── One-shot examples injected into Pool B regeneration prompt ────────────────
# Demonstrates that constraints must be fully preserved even when wording changes.

POOL_B_ONESHOT = (
    "\nHere is an example of constraint-preserving rephrasing:\n"
    "  Original: 'Summarize the article without using the words \"said\" or \"reported\".'\n"
    "  Good rephrase: 'Provide a summary of the article, avoiding the terms \"said\" and \"reported\".'\n"
    "  Bad rephrase:  'Write a brief summary of the article.'  ← constraint was dropped!\n\n"
    "Ensure the constraint is explicitly present and enforceable in your rephrased version.\n"
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_seeds(path: Path) -> Dict[str, Dict[str, Any]]:
    seeds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                seeds[s["prompt_id"]] = s
    return seeds


# ── Tighter system prompt for Pool B retry ────────────────────────────────────
# The one-shot example lives here (system prompt), NOT appended to base_text —
# base_text is also used as the reference for length-ratio/SBERT-similarity/
# dedup checks in paraphrase_generator.generate_paraphrases(), so appending the
# ~500-char example to it inflated the reference length and made every
# candidate fail length_ratio_oob (100% rejection, confirmed 2026-07-05: B01/
# B02/B05 all returned 0/5 variants before this fix).

def _pool_b_retry_system_prompt() -> str:
    return (
        "You are a prompt rewriting assistant. You are rewriting a summarization "
        "instruction that contains a SPECIFIC CONSTRAINT. You MUST preserve that "
        "constraint EXACTLY — do not soften, generalize, drop, or paraphrase the "
        "constraint away. The constraint must be explicit, enforceable, and clearly "
        "readable in your rephrased version. Return ONLY the rephrased instruction."
        + POOL_B_ONESHOT
    )


# ── Regeneration ─────────────────────────────────────────────────────────────

def regenerate_seed(
    seed: Dict[str, Any],
    generator,
    n_variants: int,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    """Re-run generation for a flagged seed. Returns new output record or None."""
    pid      = seed["prompt_id"]
    pool     = seed["pool"]
    base     = seed["base_text"]
    base_prompt = seed["base_prompt"]

    # Override the system prompt on the generator's tokenizer-level message
    # by temporarily patching SUMM_SYSTEM_PROMPTS for pool B. base_text stays
    # unmodified — it's also the reference for length-ratio/similarity/dedup
    # checks downstream, so it must match what the LLM is actually asked to
    # rephrase.
    original_b_prompt = None
    if pool == "B":
        from gensens.scripts.paraphrase_generator import SUMM_SYSTEM_PROMPTS
        original_b_prompt = SUMM_SYSTEM_PROMPTS.get("B", "")
        SUMM_SYSTEM_PROMPTS["B"] = _pool_b_retry_system_prompt()

    record_for_gen = {
        "instance_id": pid,
        "task":        seed["task"],
        "base_text":   base,
        "base_prompt": base_prompt,
        "pool":        pool,
        "metadata":    {
            "prompt_id":  pid,
            "pool":       pool,
            "dimension":  seed["dimension"],
            "complexity": seed["complexity"],
            "dataset":    seed["dataset"],
            "regeneration": True,
        },
    }

    best_variants: List[Dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        logger.info(f"  [{pid}] Regeneration attempt {attempt}/{max_attempts}")
        t0 = time.time()
        variants = generator.generate_paraphrases(
            record_for_gen,
            n_variants=n_variants,
            max_retries=2,
        )
        elapsed = time.time() - t0

        if len(variants) > len(best_variants):
            best_variants = variants

        if len(best_variants) >= n_variants:
            logger.info(
                f"  [{pid}] Regeneration succeeded on attempt {attempt} "
                f"({len(best_variants)} variants in {elapsed:.1f}s)"
            )
            break
        logger.warning(
            f"  [{pid}] attempt {attempt}: only {len(variants)} variants, retrying."
        )

    # Restore patched prompt
    if original_b_prompt is not None:
        from gensens.scripts.paraphrase_generator import SUMM_SYSTEM_PROMPTS
        SUMM_SYSTEM_PROMPTS["B"] = original_b_prompt

    base_text_clean = seed["base_text"]
    for v in best_variants:
        pt = v["paraphrased_text"]
        if base_text_clean in base_prompt:
            v["full_prompt"] = base_prompt.replace(base_text_clean, pt, 1)
        else:
            v["full_prompt"] = pt + "\n\n" + base_prompt

    still_failed = len(best_variants) < n_variants
    mean_sim = float(np.mean([v["sbert_similarity"] for v in best_variants])) if best_variants else 0.0

    logger.info(
        f"  [{pid}] Regeneration done: {len(best_variants)}/{n_variants} variants, "
        f"mean_sim={mean_sim:.3f}, unresolved={still_failed}"
    )

    return {
        "prompt_id":                   pid,
        "pool":                        pool,
        "dimension":                   seed["dimension"],
        "complexity":                  seed["complexity"],
        "base_text":                   base_text_clean,
        "base_prompt":                 base_prompt,
        "task":                        seed["task"],
        "dataset":                     seed["dataset"],
        "variants":                    best_variants,
        "n_variants_generated":        len(best_variants),
        "generation_time_s":           -1.0,
        "constraint_leakage_unresolved": still_failed and pool == "B",
        "_regenerated":                True,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    val_path    = Path(args.validation)
    data_path   = Path(args.data)
    seeds_path  = Path(args.seeds)
    out_path    = Path(args.output) if args.output else data_path

    if not val_path.exists():
        logger.error(f"Validation file not found: {val_path}")
        sys.exit(1)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    with open(val_path) as f:
        validation = json.load(f)

    flagged_ids = [
        pid for pid, v in validation["per_seed"].items() if v["flagged"]
    ]
    if not flagged_ids:
        logger.info("No flagged seeds — nothing to regenerate.")
        return
    logger.info(f"Flagged seeds ({len(flagged_ids)}): {', '.join(sorted(flagged_ids))}")

    records_list = load_jsonl(data_path)
    records = {r["prompt_id"]: r for r in records_list}
    seeds   = load_seeds(seeds_path)

    gen_kw: Dict[str, Any] = {"model_backend": args.backend, "model_id": args.model_id}
    from gensens.scripts.paraphrase_generator import ParaphraseGenerator
    logger.info("Initialising ParaphraseGenerator …")
    generator = ParaphraseGenerator(**gen_kw)
    logger.info("Generator ready.")

    n_resolved = 0
    n_unresolved = 0

    for pid in sorted(flagged_ids):
        if pid not in seeds:
            logger.warning(f"  {pid} not found in seed file — skipping.")
            continue

        new_rec = regenerate_seed(
            seeds[pid],
            generator,
            n_variants=args.n_variants,
            max_attempts=args.max_attempts,
        )
        if new_rec is None:
            logger.error(f"  {pid}: regeneration returned None — keeping original.")
            continue

        records[pid] = new_rec
        if new_rec.get("constraint_leakage_unresolved"):
            n_unresolved += 1
        else:
            n_resolved += 1

    # Write updated data file (preserve original seed ordering)
    seed_order = {s["prompt_id"]: i for i, s in enumerate(load_jsonl(seeds_path))}
    final_records = sorted(records.values(), key=lambda r: seed_order.get(r["prompt_id"], 999))
    write_jsonl(out_path, final_records)
    logger.info(
        f"Updated data written: {out_path}  "
        f"(resolved={n_resolved}, unresolved={n_unresolved})"
    )

    # Re-run validation
    if not args.skip_validation:
        logger.info("Re-running validation …")
        import importlib, gensens.scripts.validate_summ_paraphrases as val_mod
        importlib.reload(val_mod)  # ensure fresh state

        new_validation = val_mod.validate(
            final_records,
            expected_n=args.n_variants,
            skip_llm=True,        # skip LLM judge in re-validation pass
            llm_backend=args.backend,
            model_id=args.model_id,
        )

        val_path_new = val_path.with_name(val_path.stem + "_post_regen.json")
        with open(val_path_new, "w") as f:
            json.dump(new_validation, f, indent=2)
        logger.info(f"Post-regeneration validation written: {val_path_new}")

        before = validation["summary"]["flagged_seeds"]
        after  = new_validation["summary"]["flagged_seeds"]
        logger.info(
            f"\n─── Regeneration diff ───────────────────────\n"
            f"  Flagged before: {before}\n"
            f"  Flagged after:  {after}\n"
            f"  Resolved:       {n_resolved}\n"
            f"  Still flagged:  {n_unresolved}\n"
            f"────────────────────────────────────────────\n"
        )
    else:
        logger.info("Validation skipped (--skip-validation).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate flagged seed paraphrases.")
    p.add_argument("--validation",      default=str(DEFAULT_VALIDATION))
    p.add_argument("--data",            default=str(DEFAULT_DATA))
    p.add_argument("--seeds",           default=str(DEFAULT_SEEDS))
    p.add_argument("--output",          default=None,
                   help="Output path (default: overwrite --data file in-place)")
    p.add_argument("--backend",         default="vllm", choices=["vllm", "llama"])
    p.add_argument("--model-id",        default=DEFAULT_MODEL_ID)
    p.add_argument("--max-attempts",    type=int, default=MAX_ATTEMPTS)
    p.add_argument("--n-variants",      type=int, default=5)
    p.add_argument("--skip-validation", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
