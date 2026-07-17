"""
fix_flagged_seeds.py
====================
Targeted, surgical repair of the existing 50-seed summarization paraphrase
dataset — WITHOUT regenerating the whole pool. Three fixes:

  FIX 1 — Top up under-target seeds to exactly 5 variants:
          A03, A15, A18 (pool A), B05 (pool B).
  FIX 2 — Fully regenerate constraint-drift Pool B seeds:
          B14 (must convey explicit skepticism/doubt, not mere "critical"),
          B15 (must keep BOTH parts: 50-word summary + self-critique) —
          B15 is regenerated only if <4/5 current variants hold both parts.
  FIX 3 — Replace every variant (any seed) with sbert_similarity > 0.92,
          using stronger transformation strategies, keeping count at 5.

NEW acceptance band for all (re)generated variants:  0.82 <= sim <= 0.92
(the upper bound 0.92 is new; the old pipeline had only the 0.82 lower gate).
The existing bidirectional NLI gate is retained.

Everything not touched is preserved byte-for-byte: untouched seeds are copied
from the original file line verbatim; only affected records are re-serialised.

Outputs:
  data/gensens_summ_50seed_5para.jsonl           (overwritten in place)
  data/gensens_summ_50seed_5para.backup.jsonl    (pre-fix backup)
  data/fix_changelog.json                        (per-seed change log)

Usage:
    python gensens/scripts/fix_flagged_seeds.py [--model-id ...] [--backend vllm]
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
sys.path.insert(0, str(_GENSENS_ROOT.parent))  # repo root for `gensens.scripts...`

from gensens.scripts.paraphrase_generator import (  # noqa: E402
    ParaphraseGenerator, STRATEGY_NAMES, STRATEGY_INSTRUCTIONS, STRATEGY_FAMILY,
)
import gensens.scripts.validate_summ_paraphrases as val_mod  # noqa: E402

DATA_DIR   = _GENSENS_ROOT / "data"
DATA_FILE  = DATA_DIR / "gensens_summ_50seed_5para.jsonl"
BACKUP_FILE = DATA_DIR / "gensens_summ_50seed_5para.backup.jsonl"
CHANGELOG  = DATA_DIR / "fix_changelog.json"

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ── Acceptance band (FIX: upper bound 0.92 is new) ────────────────────────────
SIM_LOW  = 0.82
SIM_HIGH = 0.92
TARGET_VARIANTS = 5
MAX_RETRIES_PER_STRATEGY = 3
TEMPERATURE = 0.85
TOP_P = 0.92
MAX_NEW_TOKENS = 256

UNDER_TARGET   = ["A03", "A15", "A18", "B05"]
B14, B15 = "B14", "B15"

# Strategy → instruction lookup.
STRAT_INSTR = dict(zip(STRATEGY_NAMES, STRATEGY_INSTRUCTIONS))

# FIX 3 preferred families/strategies (stronger transforms first);
# reordered_clauses is deliberately excluded (it yields near-identical text).
FIX3_STRATEGY_ORDER = [
    "different_structure",   # syntactic — strongest restructuring
    "technical_vocab",       # lexical
    "role_prefix",           # pragmatic
    "simple_vocab",          # lexical
    "synonyms",              # lexical
    "split_sentences",       # syntactic
    "formal_tone",           # lexical
    "different_opening",     # syntactic
    "passive_voice",         # syntactic
    "casual_tone",           # lexical
    "elaborated",            # length
    "merged_sentences",      # syntactic
    "imperative",            # pragmatic
]

# General top-up / regen strategy order (diverse; reordered_clauses last).
GENERAL_STRATEGY_ORDER = [
    "different_structure", "technical_vocab", "synonyms", "simple_vocab",
    "formal_tone", "casual_tone", "role_prefix", "split_sentences",
    "different_opening", "elaborated", "merged_sentences", "imperative",
    "passive_voice", "concise", "reordered_clauses",
]


# ── Strict constraint checks (stricter than validate_summ_paraphrases) ────────

def _has(text: str, kws: List[str]) -> bool:
    tl = text.lower()
    return any(k.lower() in tl for k in kws)


def b05_constraint_ok(text: str) -> bool:
    """B05: perspective of the most negatively impacted party."""
    return _has(text, ["perspective", "point of view", "viewpoint", "from the angle", "vantage"]) \
        and _has(text, ["negative", "suffered", "harmed", "affected", "most impacted",
                        "hardest hit", "worst affected", "adversely"])


def b14_constraint_ok(text: str) -> bool:
    """B14: explicit skepticism/doubt — NOT satisfied by 'critical'/'objective' alone."""
    return _has(text, ["skeptic", "sceptic", "doubt", "dubious", "incredulous",
                       "distrust", "question the validity", "question the truth",
                       "cast doubt", "unconvinced", "suspicion", "suspect the"])


def b15_constraint_ok(text: str) -> bool:
    """B15: must retain BOTH parts — 50-word summary AND self-critique of nuance."""
    critique = _has(text, ["critique", "critiquing", "self-critique", "self-critical",
                           "criticise your", "criticize your"])
    fifty    = _has(text, ["50-word", "50 word", "fifty-word", "fifty word", "brief summary"])
    nuance   = _has(text, ["nuance", "missed", "failed to capture", "shortcoming",
                           "overlooked", "left out", "not capture"])
    return critique and fifty and nuance


# Per-seed constraint gate for (re)generated variants. Seeds not listed here
# (pool A, or pool B seeds only touched by FIX 3) fall back to the validator's
# POOL_B_CHECKS when pool == B, or no constraint when pool == A.
STRICT_CONSTRAINT = {
    "B05": b05_constraint_ok,
    "B14": b14_constraint_ok,
    "B15": b15_constraint_ok,
}


def constraint_ok(pid: str, pool: str, text: str) -> bool:
    """Return True if `text` preserves seed `pid`'s constraint."""
    if pid in STRICT_CONSTRAINT:
        return STRICT_CONSTRAINT[pid](text)
    if pool == "B":
        check = val_mod.POOL_B_CHECKS.get(pid)
        return bool(check(text)) if check else True
    return True  # pool A — no constraint


# ── One-shot system prompts for FIX 2 (constraint-preserving) ─────────────────

_POOL_B_SYS = (
    "You are a prompt rewriting assistant. Your job is to rephrase a "
    "summarization instruction that contains a SPECIFIC CONSTRAINT while "
    "keeping that constraint EXACTLY intact — do not soften, generalize, or "
    "drop it. Return ONLY the rephrased instruction, nothing else.\n"
)

B14_ONESHOT = _POOL_B_SYS + (
    "\nThe constraint here is a HIGHLY SKEPTICAL tone: the rephrase must express "
    "explicit doubt about the truthfulness of the article's claims — not merely "
    "a 'critical' or 'objective' analysis. Use an explicit skepticism/doubt "
    "marker (e.g. skeptical, doubt, dubious, incredulous, question the validity).\n"
    "\nExample:\n"
    "  ORIGINAL: 'Summarize the article in a highly skeptical tone, noting "
    "missing evidence.'\n"
    "  REPHRASED: 'Provide a doubt-filled, skeptical summary of the article's "
    "claims, calling out where the author fails to supply evidence.'\n"
)

B15_ONESHOT = _POOL_B_SYS + (
    "\nThe constraint here has TWO required parts that BOTH must survive: "
    "(1) a ~50-word summary of the article, AND (2) a second, self-critique "
    "paragraph that points out the nuance the summary missed. Keep both.\n"
    "\nExample:\n"
    "  ORIGINAL: 'First, provide a standard 50-word summary of the article. "
    "Then, write a second paragraph critiquing your own summary, pointing out "
    "what nuance you failed to capture.'\n"
    "  REPHRASED: 'Begin with a 50-word summary of the article; then add a "
    "second paragraph that critiques your own summary and identifies the "
    "nuance it missed.'\n"
)


# ── Generation primitives ─────────────────────────────────────────────────────

def _build_prompt(gen: ParaphraseGenerator, system_content: str,
                  strategy_instruction: str, base_text: str) -> str:
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",
         "content": f"{strategy_instruction}\n\nOriginal:\n{base_text}\n\nRephrased version:"},
    ]
    return gen.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _gen_one(gen: ParaphraseGenerator, system_content: str,
             strategy_instruction: str, base_text: str, seed: int) -> Optional[str]:
    """One candidate via vLLM with task sampling params + a per-call seed
    (so retries actually differ — generate_batch hardcodes seed=42)."""
    prompt = _build_prompt(gen, system_content, strategy_instruction, base_text)
    sp = gen._SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS, seed=seed,
    )
    outs = gen.llm.generate([prompt], sp)
    return gen._clean_output(outs[0].outputs[0].text) or None


def _seed_int(pid: str, strategy: str, attempt: int) -> int:
    return abs(hash((pid, strategy, attempt))) % (2**31 - 1)


def _make_variant(idx: int, strategy: str, text: str, sim: float,
                  p_fwd: Optional[float], p_bwd: Optional[float],
                  base_text: str, base_prompt: str, fix_type: str,
                  gen: ParaphraseGenerator) -> Dict[str, Any]:
    if base_text in base_prompt:
        full_prompt = base_prompt.replace(base_text, text, 1)
    else:
        full_prompt = text + "\n\n" + base_prompt
    return {
        "variant_idx": idx,
        "strategy": strategy,
        "strategy_family": STRATEGY_FAMILY.get(strategy, "lexical"),
        "paraphrased_text": text,
        "full_prompt": full_prompt,
        "sbert_similarity": round(sim, 4),
        "length_ratio": round(gen._length_ratio(text, base_text), 4),
        "token_overlap_to_base": round(gen._token_overlap(text, base_text), 4),
        "nli_entail_fwd": round(p_fwd, 4) if p_fwd is not None else None,
        "nli_entail_bwd": round(p_bwd, 4) if p_bwd is not None else None,
        "nli_passed": True,
        "best_of_n_index": -1,
        "_fixed": True,
        "_fix_type": fix_type,
    }


def generate_valid_variant(
    gen: ParaphraseGenerator,
    pid: str,
    pool: str,
    base_text: str,
    base_prompt: str,
    system_content: str,
    strategy_order: List[str],
    kept_texts: List[str],
    kept_sigs: set,
    fix_type: str,
    next_idx: int,
) -> Optional[Dict[str, Any]]:
    """Try strategies (each up to MAX_RETRIES) until one candidate passes:
    band [0.82,0.92] + NLI + not-a-base-copy + not-a-dup + constraint.
    Returns a variant dict or None if nothing qualified."""
    for strategy in strategy_order:
        instr = STRAT_INSTR[strategy]
        for attempt in range(1, MAX_RETRIES_PER_STRATEGY + 1):
            seed = _seed_int(pid, strategy, attempt)
            try:
                cand = _gen_one(gen, system_content, instr, base_text, seed)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"    [{pid}] gen error ({strategy} a{attempt}): {e}")
                continue
            if not cand:
                continue
            if cand.strip().lower() == base_text.strip().lower():
                continue
            sig = gen._first_n_words_sig(cand, 8)
            if sig in kept_sigs:
                continue
            if any(cand.strip().lower() == kt.strip().lower() for kt in kept_texts):
                continue
            sim = gen._compute_similarity(base_text, cand)
            if not (SIM_LOW <= sim <= SIM_HIGH):
                continue
            if not constraint_ok(pid, pool, cand):
                continue
            passes, p_fwd, p_bwd, _ = gen._bidirectional_nli(base_text, cand)
            if not passes:
                continue
            v = _make_variant(next_idx, strategy, cand, sim, p_fwd, p_bwd,
                              base_text, base_prompt, fix_type, gen)
            logger.info(f"    [{pid}] +variant [{strategy}] sim={sim:.3f} ({fix_type})")
            return v
    return None


# ── Fix routines ──────────────────────────────────────────────────────────────

def fix_topup(gen, rec, changelog):
    pid, pool = rec["prompt_id"], rec["pool"]
    base_text, base_prompt = rec["base_text"], rec["base_prompt"]
    variants = list(rec["variants"])
    old_n = len(variants)
    old_sim = float(np.mean([v["sbert_similarity"] for v in variants])) if variants else 0.0

    kept_texts = [v["paraphrased_text"] for v in variants]
    kept_sigs = {gen._first_n_words_sig(t, 8) for t in kept_texts}
    used_strats = {v.get("strategy") for v in variants}
    order = [s for s in GENERAL_STRATEGY_ORDER if s not in used_strats] + \
            [s for s in GENERAL_STRATEGY_ORDER if s in used_strats]

    logger.info(f"[{pid}] FIX1 top-up: {old_n} -> {TARGET_VARIANTS} (pool {pool})")
    while len(variants) < TARGET_VARIANTS:
        v = generate_valid_variant(
            gen, pid, pool, base_text, base_prompt, SUMM_SYS(pool),
            order, kept_texts, kept_sigs, "topup", len(variants),
        )
        if v is None:
            logger.warning(f"[{pid}] top-up could not reach target; keeping {len(variants)}.")
            break
        variants.append(v)
        kept_texts.append(v["paraphrased_text"])
        kept_sigs.add(gen._first_n_words_sig(v["paraphrased_text"], 8))

    _reindex(variants)
    rec["variants"] = variants
    rec["n_variants_generated"] = len(variants)
    changelog[pid] = {
        "action": "topped_up",
        "old_variant_count": old_n, "new_variant_count": len(variants),
        "old_mean_sim": round(old_sim, 4),
        "new_mean_sim": round(float(np.mean([v["sbert_similarity"] for v in variants])), 4),
        "unresolved": len(variants) < TARGET_VARIANTS,
    }


def fix_regen(gen, rec, changelog, oneshot, strict_check, force):
    pid, pool = rec["prompt_id"], rec["pool"]
    base_text, base_prompt = rec["base_text"], rec["base_prompt"]
    old_variants = list(rec["variants"])
    old_n = len(old_variants)
    old_sim = float(np.mean([v["sbert_similarity"] for v in old_variants])) if old_variants else 0.0

    if not force:
        n_hold = sum(1 for v in old_variants if strict_check(v["paraphrased_text"]))
        logger.info(f"[{pid}] FIX2 verify: {n_hold}/{old_n} hold both parts")
        if n_hold >= 4:
            changelog[pid] = {
                "action": "verified_kept", "old_variant_count": old_n,
                "new_variant_count": old_n, "old_mean_sim": round(old_sim, 4),
                "new_mean_sim": round(old_sim, 4), "unresolved": False,
                "note": f"{n_hold}/{old_n} held both parts (>=4), not regenerated",
            }
            return

    logger.info(f"[{pid}] FIX2 full regenerate (pool {pool})")
    variants: List[Dict[str, Any]] = []
    kept_texts: List[str] = []
    kept_sigs: set = set()
    order = list(GENERAL_STRATEGY_ORDER)
    while len(variants) < TARGET_VARIANTS:
        v = generate_valid_variant(
            gen, pid, pool, base_text, base_prompt, oneshot,
            order, kept_texts, kept_sigs, "regenerated", len(variants),
        )
        if v is None:
            logger.warning(f"[{pid}] regen could not reach target; keeping {len(variants)}.")
            break
        variants.append(v)
        kept_texts.append(v["paraphrased_text"])
        kept_sigs.add(gen._first_n_words_sig(v["paraphrased_text"], 8))

    if len(variants) < TARGET_VARIANTS:
        # Fallback: backfill from any OLD variants that still pass the strict
        # check and the new band, rather than shipping an under-target seed.
        for ov in old_variants:
            if len(variants) >= TARGET_VARIANTS:
                break
            t = ov["paraphrased_text"]
            sig = gen._first_n_words_sig(t, 8)
            if sig in kept_sigs:
                continue
            if strict_check(t) and SIM_LOW <= ov["sbert_similarity"] <= SIM_HIGH:
                ov2 = dict(ov)
                variants.append(ov2)
                kept_sigs.add(sig)
                logger.info(f"    [{pid}] backfilled 1 old variant that still qualifies")

    _reindex(variants)
    rec["variants"] = variants
    rec["n_variants_generated"] = len(variants)
    changelog[pid] = {
        "action": "regenerated",
        "old_variant_count": old_n, "new_variant_count": len(variants),
        "old_mean_sim": round(old_sim, 4),
        "new_mean_sim": round(float(np.mean([v["sbert_similarity"] for v in variants])), 4) if variants else 0.0,
        "unresolved": len(variants) < TARGET_VARIANTS,
    }


def fix_oversimilar(gen, rec, changelog) -> bool:
    """Replace over-similar (>0.92) variants. Returns True if the record was
    modified (an over-similar variant is ALWAYS dropped when present, even if
    its replacement fails — so 'modified' must not hinge on replacement
    success, or the dropped-but-not-replaced state is silently lost on write)."""
    pid, pool = rec["prompt_id"], rec["pool"]
    base_text, base_prompt = rec["base_text"], rec["base_prompt"]
    variants = list(rec["variants"])
    over = [v for v in variants if v.get("sbert_similarity", 0.0) > SIM_HIGH]
    if not over:
        return False
    old_n = len(variants)
    old_sim = float(np.mean([v["sbert_similarity"] for v in variants]))
    logger.info(f"[{pid}] FIX3 replace {len(over)} over-similar variant(s) (pool {pool})")

    kept = [v for v in variants if v.get("sbert_similarity", 0.0) <= SIM_HIGH]
    kept_texts = [v["paraphrased_text"] for v in kept]
    kept_sigs = {gen._first_n_words_sig(t, 8) for t in kept_texts}

    new_variants = list(kept)
    n_replaced = 0
    n_unresolved = 0
    for _ in over:
        v = generate_valid_variant(
            gen, pid, pool, base_text, base_prompt, SUMM_SYS(pool),
            FIX3_STRATEGY_ORDER, kept_texts, kept_sigs, "oversim_replace",
            len(new_variants),
        )
        if v is None:
            logger.warning(f"[{pid}] over-similar replacement failed for 1 slot.")
            n_unresolved += 1
            continue
        new_variants.append(v)
        kept_texts.append(v["paraphrased_text"])
        kept_sigs.add(gen._first_n_words_sig(v["paraphrased_text"], 8))
        n_replaced += 1

    _reindex(new_variants)
    rec["variants"] = new_variants
    rec["n_variants_generated"] = len(new_variants)
    prev = changelog.get(pid, {})
    changelog[pid] = {
        **prev,
        "action": (prev.get("action") + "+oversim_replaced") if prev.get("action") else "oversim_replaced",
        "old_variant_count": prev.get("old_variant_count", old_n),
        "new_variant_count": len(new_variants),
        "old_mean_sim": prev.get("old_mean_sim", round(old_sim, 4)),
        "new_mean_sim": round(float(np.mean([v["sbert_similarity"] for v in new_variants])), 4) if new_variants else 0.0,
        "n_oversimilar_replaced": n_replaced,
        "unresolved": bool(prev.get("unresolved")) or n_unresolved > 0 or len(new_variants) < TARGET_VARIANTS,
    }
    return True  # an over-similar variant was present and dropped/replaced


def _reindex(variants: List[Dict[str, Any]]) -> None:
    for i, v in enumerate(variants):
        v["variant_idx"] = i


def SUMM_SYS(pool: str) -> str:
    from gensens.scripts.paraphrase_generator import SUMM_SYSTEM_PROMPTS
    return SUMM_SYSTEM_PROMPTS.get(pool, SUMM_SYSTEM_PROMPTS["A"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Read raw lines (to preserve untouched records byte-for-byte) + parse.
    raw_lines: List[str] = []
    records: Dict[str, Dict[str, Any]] = {}
    order_ids: List[str] = []
    with open(DATA_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            raw_lines.append(line.rstrip("\n"))
            rec = json.loads(line)
            records[rec["prompt_id"]] = rec
            order_ids.append(rec["prompt_id"])
    raw_by_id = {pid: raw for pid, raw in zip(order_ids, raw_lines)}
    logger.info(f"Loaded {len(records)} seeds from {DATA_FILE}")

    logger.info(f"Initialising ParaphraseGenerator ({args.backend}, {args.model_id}) …")
    gen = ParaphraseGenerator(
        model_backend=args.backend,
        model_id=args.model_id,
        similarity_threshold=SIM_LOW,
        enable_nli_gate=True,
    )
    logger.info("Generator ready.")

    changelog: Dict[str, Any] = {}
    touched: set = set()

    # FIX 1 — top up under-target seeds.
    for pid in UNDER_TARGET:
        try:
            fix_topup(gen, records[pid], changelog)
            touched.add(pid)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{pid}] FIX1 exception: {e}", exc_info=True)

    # FIX 2 — regenerate constraint-drift seeds. B14 forced; B15 conditional.
    try:
        fix_regen(gen, records[B14], changelog, B14_ONESHOT, b14_constraint_ok, force=True)
        touched.add(B14)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{B14}] FIX2 exception: {e}", exc_info=True)
    try:
        fix_regen(gen, records[B15], changelog, B15_ONESHOT, b15_constraint_ok, force=False)
        if changelog.get(B15, {}).get("action") != "verified_kept":
            touched.add(B15)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{B15}] FIX2 exception: {e}", exc_info=True)

    # FIX 3 — replace over-similar variants across ALL seeds (current state).
    # Seeds fully regenerated in FIX 2 are already within [0.82, 0.92], so their
    # (post-regen) variants carry no over-similar entries; scanning them is a
    # harmless no-op but we skip explicitly for clarity.
    for pid in order_ids:
        if changelog.get(pid, {}).get("action") == "regenerated":
            continue
        try:
            # Mark touched whenever an over-similar variant was present (it is
            # dropped even if not replaced) — not only on successful replace.
            if fix_oversimilar(gen, records[pid], changelog):
                touched.add(pid)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{pid}] FIX3 exception: {e}", exc_info=True)

    # ── Backup then write merged output ──────────────────────────────────────
    shutil.copyfile(DATA_FILE, BACKUP_FILE)
    logger.info(f"Backed up original -> {BACKUP_FILE}")

    with open(DATA_FILE, "w") as f:
        for pid in order_ids:
            if pid in touched:
                f.write(json.dumps(records[pid], ensure_ascii=False) + "\n")
            else:
                f.write(raw_by_id[pid] + "\n")
    logger.info(f"Wrote merged dataset ({len(touched)} seeds modified): {sorted(touched)}")

    with open(CHANGELOG, "w") as f:
        json.dump(changelog, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote changelog -> {CHANGELOG}")

    # ── Final validation report ──────────────────────────────────────────────
    final_report(records, order_ids)


def final_report(records, order_ids):
    logger.info("\n" + "=" * 62)
    logger.info("FINAL VALIDATION REPORT")
    logger.info("=" * 62)

    all_five = True
    band_ok = True
    n_variants_total = 0
    below, above = [], []
    base_copies = []
    missing_article = []
    constraint_fail = []

    for pid in order_ids:
        rec = records[pid]
        pool = rec["pool"]
        vs = rec["variants"]
        n_variants_total += len(vs)
        if len(vs) != TARGET_VARIANTS:
            all_five = False
            logger.warning(f"  {pid}: {len(vs)} variants (expected {TARGET_VARIANTS})")
        for v in vs:
            s = v["sbert_similarity"]
            if s < SIM_LOW:
                below.append((pid, v["variant_idx"], s)); band_ok = False
            if s > SIM_HIGH:
                above.append((pid, v["variant_idx"], s)); band_ok = False
            if v["paraphrased_text"].strip().lower() == rec["base_text"].strip().lower():
                base_copies.append((pid, v["variant_idx"]))
            if "{{article}}" not in v["full_prompt"]:
                missing_article.append((pid, v["variant_idx"]))
        if pool == "B":
            n_ok = sum(1 for v in vs if constraint_ok(pid, pool, v["paraphrased_text"]))
            if n_ok < 4:
                constraint_fail.append((pid, n_ok, len(vs)))

    logger.info(f"1. All 50 seeds exactly 5 variants: {'PASS' if all_five else 'FAIL'}")
    logger.info(f"2. All variants in [{SIM_LOW}, {SIM_HIGH}]: "
                f"{'PASS' if band_ok else 'FAIL'} "
                f"(below={len(below)}, above={len(above)})")
    if below: logger.warning(f"     below 0.82: {below}")
    if above: logger.warning(f"     above 0.92: {above}")
    logger.info(f"3. Pool B constraint >=4/5 each: "
                f"{'PASS' if not constraint_fail else 'FAIL'}")
    if constraint_fail:
        logger.warning(f"     under-threshold: {constraint_fail}")
    # explicit B14 / B15 strict re-check
    b14_ok = sum(1 for v in records[B14]["variants"] if b14_constraint_ok(v["paraphrased_text"]))
    b15_ok = sum(1 for v in records[B15]["variants"] if b15_constraint_ok(v["paraphrased_text"]))
    logger.info(f"     B14 skepticism markers: {b14_ok}/{len(records[B14]['variants'])}")
    logger.info(f"     B15 both-parts:         {b15_ok}/{len(records[B15]['variants'])}")
    logger.info(f"4. No base_text copies: {'PASS' if not base_copies else 'FAIL'} "
                f"| all full_prompts have {{{{article}}}}: {'PASS' if not missing_article else 'FAIL'}")
    if base_copies: logger.warning(f"     copies: {base_copies}")
    if missing_article: logger.warning(f"     missing article ph: {missing_article}")

    # Summary table
    logger.info("\n5. Summary table")
    logger.info(f"   {'Pool':<6}{'Seeds':>6}{'Variants':>10}{'MeanSBERT':>11}{'Min':>8}{'Max':>8}{'Complete':>10}")
    for pool in ("A", "B"):
        recs = [records[p] for p in order_ids if records[p]["pool"] == pool]
        sims = [v["sbert_similarity"] for r in recs for v in r["variants"]]
        tv = sum(len(r["variants"]) for r in recs)
        complete = all(len(r["variants"]) == TARGET_VARIANTS for r in recs)
        logger.info(f"   {pool:<6}{len(recs):>6}{tv:>10}{np.mean(sims):>11.4f}"
                    f"{np.min(sims):>8.4f}{np.max(sims):>8.4f}{str(complete):>10}")
    logger.info(f"   Total variants: {n_variants_total}")
    logger.info("=" * 62)


def parse_args():
    p = argparse.ArgumentParser(description="Surgically fix flagged seeds in the paraphrase dataset.")
    p.add_argument("--backend", default="vllm", choices=["vllm", "llama"])
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
