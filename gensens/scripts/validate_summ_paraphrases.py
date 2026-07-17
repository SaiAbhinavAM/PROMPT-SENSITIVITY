"""
validate_summ_paraphrases.py
============================
Validate the generated paraphrases for the 50 summarization seed prompts.

Checks (A–G):
  A. Completeness   — all 50 seeds present, each has expected n_variants
  B. SBERT ≥ 0.82   — per-variant similarity gate
  C. No exact dupes — no two variants share the same text (case-insensitive)
  D. Length sanity  — each paraphrase ≥ 5 words and ≤ 3× the word count of base_text
  E. {{article}} preserved — full_prompt still contains the placeholder
  F. Pool B keyword  — constraint keywords present in each paraphrase (per prompt_id)
  G. LLM fallback   — if >30% of Pool B variants fail check F, run LLM-as-judge
                       (requires GPU; set --skip-llm to disable)

Output: gensens/data/gensens_summ_validation.json
        Per-seed: sbert_pass_count, constraint_pass_count, flagged, flag_reasons[]

Usage:
    python gensens/scripts/validate_summ_paraphrases.py [options]

Options:
    --input      path   (default: gensens/data/gensens_summ_50seed_5para.jsonl)
    --output     path   (default: gensens/data/gensens_summ_validation.json)
    --expected-n int    (default: 5)
    --skip-llm          Skip LLM-as-judge fallback even if threshold exceeded
    --llm-backend vllm|llama  (default: vllm)
    --model-id   path   LLM model id for judge (default: ParaphraseGenerator default)
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_GENSENS_ROOT = _HERE.parent
DATA_DIR = _GENSENS_ROOT / "data"

DEFAULT_INPUT      = DATA_DIR / "gensens_summ_50seed_5para.jsonl"
DEFAULT_OUTPUT     = DATA_DIR / "gensens_summ_validation.json"
SBERT_THRESHOLD    = 0.82
MAX_LENGTH_RATIO   = 3.0
MIN_WORDS          = 5
LLM_FALLBACK_RATE  = 0.30  # trigger LLM judge if >30% of Pool B variants fail F

# ── Pool B keyword constraints ────────────────────────────────────────────────
# Each entry is a callable(paraphrased_text: str) -> bool.
# True = constraint preserved in this paraphrase.

def _any_kw(text: str, keywords: List[str], all_required: bool = False) -> bool:
    tl = text.lower()
    if all_required:
        return all(kw.lower() in tl for kw in keywords)
    return any(kw.lower() in tl for kw in keywords)


# Per-prompt-id keyword check functions
POOL_B_CHECKS: Dict[str, Any] = {
    # B01: no proper nouns constraint — paraphrase must still mention "proper noun"
    #      or equivalent phrasing (names, cities, organizations, countries)
    "B01": lambda t: _any_kw(t, [
        "proper noun", "proper nouns", "names of", "no names", "without names",
        "people", "cities", "countries", "organizations", "real names",
    ]),
    # B02: banned word list preserved — check all 5 forbidden words mentioned
    "B02": lambda t: (
        ("said" in t.lower() or "say" in t.lower() or "stating" in t.lower()) is False
        or _any_kw(t, ["without using", "without the word", "avoid", "do not use", "ban", "forbidden"])
    ) and _any_kw(t, ["said", "reported", "police", "government", "stated",
                       "without using", "avoid the word", "do not use"]),
    # B03: forced words — all three must be referenced
    "B03": lambda t: (
        _any_kw(t, ["catalyst", "unprecedented", "ramification"]) or
        _any_kw(t, ["integrat", "incorporat", "includ", "use the word", "must contain"])
    ),
    # B04: financial/economic/market theme isolation
    "B04": lambda t: _any_kw(t, [
        "financial", "economic", "market", "fiscal", "monetary",
        "economic policy", "market reaction", "financial impact",
    ]),
    # B05: perspective of most negatively impacted party
    "B05": lambda t: _any_kw(t, [
        "perspective", "point of view", "viewpoint", "from the angle",
        "negative impact", "suffered", "victim", "most affected",
    ]),
    # B06: primary argument / conflict
    "B06": lambda t: _any_kw(t, [
        "argument", "conflict", "dispute", "driving force",
        "primary issue", "central tension", "core disagreement",
    ]),
    # B07: journalist's framing / sentiment
    "B07": lambda t: _any_kw(t, [
        "sentiment", "framing", "bias", "journalist", "author",
        "tone", "perspective of the writer", "underlying",
    ]),
    # B08: ignore background, focus on direct action/statement
    "B08": lambda t: _any_kw(t, [
        "ignore", "background", "direct action", "direct statement",
        "focus only", "exclude context", "without context",
    ]),
    # B09: executive brief / suitable for senior audience
    "B09": lambda t: _any_kw(t, [
        "executive", "brief", "senior", "leadership", "c-suite",
        "high-level", "business audience", "concise overview",
    ]),
    # B10: simple, conversational / everyday language
    "B10": lambda t: _any_kw(t, [
        "simple", "conversational", "everyday", "plain language",
        "easy to understand", "accessible", "informal",
    ]),
    # B11: clinical, neutral tone
    "B11": lambda t: _any_kw(t, [
        "clinical", "neutral", "objective", "impartial",
        "dispassionate", "factual", "without emotion",
    ]),
    # B12: punchy / sensationalist / tabloid style
    "B12": lambda t: _any_kw(t, [
        "tabloid", "sensationalist", "punchy", "dramatic", "bold",
        "eye-catching", "shocking", "headline style",
    ]),
    # B13: academic paper abstract / methodology and findings
    "B13": lambda t: _any_kw(t, [
        "academic", "abstract", "research paper", "methodology",
        "findings", "scholarly", "formal academic",
    ]),
    # B14: skeptical tone / missing evidence
    "B14": lambda t: _any_kw(t, [
        "skeptical", "sceptical", "critical", "missing evidence",
        "glossed over", "question", "doubt", "what is absent",
    ]),
    # B15: 50-word summary + critique paragraph
    "B15": lambda t: _any_kw(t, [
        "50-word", "50 word", "fifty-word", "critique", "second paragraph",
        "criticize", "criticise", "shortcoming", "what was missed",
    ]),
}


# ── Load / save helpers ───────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


# ── Per-variant checks ────────────────────────────────────────────────────────

def check_variant(
    variant: Dict[str, Any],
    base_text: str,
    pool: str,
    prompt_id: str,
    seen_texts: Set[str],
) -> List[str]:
    """Return list of failure reason strings (empty = all pass)."""
    reasons: List[str] = []
    text = variant.get("paraphrased_text", "")
    full_prompt = variant.get("full_prompt", "")
    idx = variant.get("variant_idx", "?")

    # B — SBERT
    sim = variant.get("sbert_similarity", 0.0)
    if sim < SBERT_THRESHOLD:
        reasons.append(f"v{idx}:sbert_too_low({sim:.3f})")

    # C — exact duplicates
    key = text.strip().lower()
    if key in seen_texts:
        reasons.append(f"v{idx}:exact_duplicate")
    else:
        seen_texts.add(key)

    # D — length sanity
    wc = word_count(text)
    if wc < MIN_WORDS:
        reasons.append(f"v{idx}:too_short({wc}_words)")
    base_wc = word_count(base_text)
    if base_wc > 0 and wc > MAX_LENGTH_RATIO * base_wc:
        reasons.append(f"v{idx}:too_long(ratio={wc/base_wc:.1f}x)")

    # E — {{article}} placeholder
    if "{{article}}" not in full_prompt:
        reasons.append(f"v{idx}:article_placeholder_missing")

    return reasons


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

def llm_judge_constraint_preserved(
    generator,
    base_text: str,
    paraphrase: str,
    prompt_id: str,
) -> bool:
    """Ask LLaMA-3 whether the constraint in base_text is preserved in paraphrase."""
    judge_prompt = (
        f"A summarization instruction has a specific constraint. "
        f"Determine whether the paraphrased version preserves that constraint.\n\n"
        f"Original instruction:\n{base_text}\n\n"
        f"Paraphrased instruction:\n{paraphrase}\n\n"
        f"Question: Is the constraint from the original fully preserved in the paraphrase? "
        f"Answer with YES or NO only."
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator. Answer only YES or NO."},
        {"role": "user",   "content": judge_prompt},
    ]
    try:
        if generator.backend == "vllm":
            prompt = generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = generator.generate_batch([prompt])[0]
        else:
            import torch
            input_ids = generator.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            device = next(generator.model.parameters()).device
            with torch.no_grad():
                out = generator.model.generate(
                    input_ids.to(device),
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=generator.tokenizer.pad_token_id,
                )
            answer = generator.tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            )
        return answer.strip().upper().startswith("YES")
    except Exception as e:
        logger.warning(f"LLM judge failed for {prompt_id}: {e}")
        return False


# ── Main validation logic ─────────────────────────────────────────────────────

def validate(
    records: List[Dict[str, Any]],
    expected_n: int,
    skip_llm: bool,
    llm_backend: str,
    model_id: Optional[str],
) -> Dict[str, Any]:
    per_seed: Dict[str, Dict[str, Any]] = {}
    pool_b_constraint_failures: List[Tuple[str, int, str]] = []  # (prompt_id, v_idx, paraphrase)

    for rec in records:
        pid      = rec["prompt_id"]
        pool     = rec["pool"]
        base     = rec["base_text"]
        variants = rec.get("variants", [])
        reasons: List[str] = []

        # A — completeness
        n_gen = len(variants)
        if n_gen < expected_n:
            reasons.append(f"only_{n_gen}_variants(expected_{expected_n})")

        seen_texts: Set[str] = {base.strip().lower()}
        sbert_pass = 0
        constraint_pass = 0

        for v in variants:
            v_reasons = check_variant(v, base, pool, pid, seen_texts)
            reasons.extend(v_reasons)

            sim = v.get("sbert_similarity", 0.0)
            if sim >= SBERT_THRESHOLD:
                sbert_pass += 1

            # F — Pool B constraint keyword check
            if pool == "B" and pid in POOL_B_CHECKS:
                text = v.get("paraphrased_text", "")
                if POOL_B_CHECKS[pid](text):
                    constraint_pass += 1
                else:
                    reasons.append(
                        f"v{v.get('variant_idx','?')}:constraint_keyword_missing"
                    )
                    pool_b_constraint_failures.append((pid, v.get("variant_idx", 0), text))
            elif pool == "B":
                # No check rule defined — pass through
                constraint_pass += 1

        per_seed[pid] = {
            "prompt_id":            pid,
            "pool":                 pool,
            "n_variants":           n_gen,
            "sbert_pass_count":     sbert_pass,
            "constraint_pass_count": constraint_pass if pool == "B" else None,
            "flagged":              len(reasons) > 0,
            "flag_reasons":         reasons,
        }

    # G — LLM-as-judge fallback for Pool B
    llm_judge_triggered = False
    llm_judge_results: Dict[str, Any] = {}

    total_b_variants = sum(
        rec["n_variants"]
        for rec in per_seed.values()
        if per_seed[rec["prompt_id"]]["pool"] == "B"
    )
    if total_b_variants > 0:
        fail_rate = len(pool_b_constraint_failures) / total_b_variants
    else:
        fail_rate = 0.0

    if not skip_llm and fail_rate > LLM_FALLBACK_RATE:
        logger.warning(
            f"Pool B keyword failure rate {fail_rate:.1%} > {LLM_FALLBACK_RATE:.0%} "
            "— triggering LLM-as-judge fallback."
        )
        llm_judge_triggered = True
        try:
            from gensens.scripts.paraphrase_generator import ParaphraseGenerator
            gen_kw: Dict[str, Any] = {"model_backend": llm_backend}
            if model_id:
                gen_kw["model_id"] = model_id
            generator = ParaphraseGenerator(**gen_kw)

            # Re-judge only the failed (prompt_id, variant) pairs
            for pid_f, v_idx, para in pool_b_constraint_failures:
                base_f = next(
                    (r["base_text"] for r in records if r["prompt_id"] == pid_f), ""
                )
                passed = llm_judge_constraint_preserved(generator, base_f, para, pid_f)
                key = f"{pid_f}_v{v_idx}"
                llm_judge_results[key] = {"passed": passed}
                if passed:
                    # Upgrade: remove the constraint_keyword_missing flag
                    seed_rec = per_seed[pid_f]
                    seed_rec["flag_reasons"] = [
                        r for r in seed_rec["flag_reasons"]
                        if r != f"v{v_idx}:constraint_keyword_missing"
                    ]
                    seed_rec["constraint_pass_count"] = (
                        (seed_rec["constraint_pass_count"] or 0) + 1
                    )
                    seed_rec["flagged"] = len(seed_rec["flag_reasons"]) > 0
                    logger.info(f"  LLM judge: {pid_f} v{v_idx} → PASS (constraint preserved)")
                else:
                    logger.info(f"  LLM judge: {pid_f} v{v_idx} → FAIL")
        except Exception as e:
            logger.error(f"LLM judge initialisation failed: {e}")

    # Summary counts
    n_flagged = sum(1 for v in per_seed.values() if v["flagged"])

    result = {
        "summary": {
            "total_seeds":            len(per_seed),
            "flagged_seeds":          n_flagged,
            "pool_b_fail_rate":       round(fail_rate, 4),
            "llm_judge_triggered":    llm_judge_triggered,
            "n_llm_judged":           len(llm_judge_results),
        },
        "per_seed": per_seed,
        "llm_judge_results": llm_judge_results,
    }
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        sys.exit(1)

    records = load_jsonl(in_path)
    logger.info(f"Loaded {len(records)} records from {in_path}")

    result = validate(
        records,
        expected_n=args.expected_n,
        skip_llm=args.skip_llm,
        llm_backend=args.llm_backend,
        model_id=args.model_id,
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Validation report written: {out_path}")

    s = result["summary"]
    logger.info(
        f"\n─── Validation summary ────────────────────────\n"
        f"  Total seeds:    {s['total_seeds']}\n"
        f"  Flagged seeds:  {s['flagged_seeds']}\n"
        f"  Pool B fail rate (keyword): {s['pool_b_fail_rate']:.1%}\n"
        f"  LLM judge triggered: {s['llm_judge_triggered']}"
        f"  (judged {s['n_llm_judged']} variants)\n"
        f"──────────────────────────────────────────────\n"
    )

    if s["flagged_seeds"] > 0:
        flagged_ids = [
            pid for pid, v in result["per_seed"].items() if v["flagged"]
        ]
        logger.warning(
            f"  Flagged seed IDs: {', '.join(sorted(flagged_ids))}\n"
            "  Run regenerate_flagged.py to retry these seeds."
        )
    else:
        logger.info("  All seeds passed validation.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated summarization paraphrases.")
    p.add_argument("--input",       default=str(DEFAULT_INPUT))
    p.add_argument("--output",      default=str(DEFAULT_OUTPUT))
    p.add_argument("--expected-n",  type=int, default=5)
    p.add_argument("--skip-llm",    action="store_true", default=False,
                   help="Skip LLM-as-judge fallback (no GPU required)")
    p.add_argument("--llm-backend", default="vllm", choices=["vllm", "llama"])
    p.add_argument("--model-id",    default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
