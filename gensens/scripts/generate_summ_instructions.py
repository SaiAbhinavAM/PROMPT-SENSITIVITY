#!/usr/bin/env python3
"""
generate_summ_instructions.py
==============================
Generates 100 unique summarization instructions across 4 complexity levels
using an open-source 8B model (default: Qwen/Qwen2.5-7B-Instruct) on H100.

Complexity levels  (25 instructions each → 100 total):
  1 - simple   : ≤12 words, plain everyday language
  2 - moderate : 12–20 words, standard with basic specificity
  3 - detailed : 20–35 words, professional with explicit content requirements
  4 - complex  : 35–60 words, elaborate multi-clause framing

Output: gensens/data/summ_instruction_pool.jsonl
  One JSON record per line:
    {
      "instruction":      str,
      "complexity_level": int,   # 1–4
      "complexity_label": str,   # "simple" / "moderate" / "detailed" / "complex"
      "sbert_similarity": float  # cosine similarity to the canonical instruction
    }

Usage:
  # HF Transformers (single GPU / multi-GPU via device_map="auto"):
  python scripts/generate_summ_instructions.py --backend hf

  # vLLM (H100, fastest):
  python scripts/generate_summ_instructions.py --backend vllm

  # Override model:
  python scripts/generate_summ_instructions.py --backend vllm \\
      --model-id Qwen/Qwen2.5-7B-Instruct

  # Custom pool size per level:
  python scripts/generate_summ_instructions.py --n-per-level 30

generate_dataset.py automatically picks up the saved pool file on the next run,
skipping on-the-fly instruction generation.
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util as st_util

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SBERT_MODEL_ID   = "all-mpnet-base-v2"

CANONICAL_INSTRUCTION = (
    "Summarize the following news article in 3-4 sentences, "
    "capturing the main events and key details."
)

SBERT_LOWER       = 0.82   # reject if similarity < this (meaning drift)
SBERT_UPPER       = 0.98   # reject if similarity > this (trivial restatement)
TOKEN_OVERLAP_MAX = 0.85   # reject if Jaccard word overlap > this
MIN_WORDS         = 6      # reject candidates shorter than this

# Level definitions: (level_id, label, generation_prompt_template)
# {n} is replaced with the requested count; model is asked to output one per line.
COMPLEXITY_LEVELS: List[Tuple[int, str, str]] = [
    (
        1, "simple",
        "Generate {n} very short, simple, direct instructions for asking someone to summarize "
        "a news article in 3-4 sentences. Use plain everyday language. "
        "Keep each instruction under 12 words. "
        "Example style: 'Summarize this article in 3-4 sentences.' "
        "Output ONLY the instructions, one per line, with no numbering, bullets, or extra text."
    ),
    (
        2, "moderate",
        "Generate {n} moderate-length instructions for summarizing a news article in 3-4 sentences. "
        "Use standard clear language with some specificity about what to include "
        "(e.g. main events, key points). Each instruction should be 12–20 words. "
        "Output ONLY the instructions, one per line, with no numbering, bullets, or extra text."
    ),
    (
        3, "detailed",
        "Generate {n} detailed instructions for summarizing a news article in 3-4 sentences. "
        "Specify which content elements to cover (main events, key figures, core findings, context). "
        "Use professional language. Each instruction should be 20–35 words. "
        "Output ONLY the instructions, one per line, with no numbering, bullets, or extra text."
    ),
    (
        4, "complex",
        "Generate {n} complex, elaborate instructions for summarizing a news article in 3-4 sentences. "
        "Use sophisticated vocabulary, multi-clause sentence structures, and precise framing about "
        "narrative arc, key stakeholders, thematic significance, and factual accuracy. "
        "Each instruction should be 35–60 words. "
        "Output ONLY the instructions, one per line, with no numbering, bullets, or extra text."
    ),
]

SYSTEM_PROMPT = (
    "You are an expert at generating diverse, well-formed task instructions. "
    "Follow the user's formatting rules exactly."
)


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_hf_model(model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    logging.info(f"Loading {model_id} via HF Transformers (bfloat16, device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    logging.info("  HF model loaded.")
    return tokenizer, model


def load_vllm_model(model_id: str, gpu_memory_utilization: float, max_model_len: int):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vllm not installed. On H100: pip install vllm") from e
    from transformers import AutoTokenizer
    logging.info(f"Loading {model_id} via vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="auto",
        trust_remote_code=True,
    )
    logging.info("  vLLM engine ready.")
    return tokenizer, llm, SamplingParams


# ─────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────

def _build_messages(user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def generate_hf(tokenizer, model, messages: List[Dict], max_new_tokens: int) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    input_len = input_ids.shape[1]
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model.generate(
            input_ids.to(device),
            max_new_tokens=max_new_tokens,
            temperature=0.85,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def generate_vllm(tokenizer, llm, SamplingParams, messages: List[Dict], max_new_tokens: int) -> str:
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sp = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=max_new_tokens, seed=42)
    return llm.generate([full_prompt], sp)[0].outputs[0].text


# ─────────────────────────────────────────────────────────────
# Parsing & filtering helpers
# ─────────────────────────────────────────────────────────────

def parse_lines(raw: str) -> List[str]:
    """Extract one instruction per line; strip leading numbers/bullets."""
    lines = raw.split("\n")
    out = []
    for line in lines:
        line = re.sub(r'^\s*\d+[.)]\s*', '', line)
        line = re.sub(r'^\s*[-*•]\s*', '', line).strip()
        if len(line.split()) >= MIN_WORDS:
            out.append(line)
    return out


def first_n_words(text: str, n: int = 8) -> str:
    return " ".join(text.lower().split()[:n])


def token_overlap(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sbert_filter(
    candidates: List[str],
    canon_emb,
    sbert: SentenceTransformer,
    seen_sigs: set,
    lower: float,
    upper: float,
    overlap_max: float,
    canonical: str,
) -> List[Dict[str, Any]]:
    """Return list of {instruction, sbert_similarity} that pass all filters."""
    accepted = []
    for cand in candidates:
        sig = first_n_words(cand)
        if sig in seen_sigs:
            continue
        emb = sbert.encode(cand, convert_to_tensor=True)
        sim = float(st_util.cos_sim(canon_emb, emb).item())
        if lower <= sim <= upper:
            if token_overlap(cand, canonical) <= overlap_max:
                accepted.append({"instruction": cand, "sbert_similarity": round(sim, 4)})
                seen_sigs.add(sig)
    return accepted


# ─────────────────────────────────────────────────────────────
# Per-level generation with retry
# ─────────────────────────────────────────────────────────────

def generate_level(
    level_id: int,
    label: str,
    prompt_template: str,
    n_target: int,
    canon_emb,
    sbert: SentenceTransformer,
    seen_sigs: set,
    canonical: str,
    backend: str,
    tokenizer,
    model_or_llm,
    SamplingParamsCls,
    max_retries: int = 4,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """Generate n_target instructions for one complexity level with retries."""
    log = logger or logging.getLogger(__name__)
    collected: List[Dict[str, Any]] = []
    ask_n = n_target  # start by asking for exactly what we need

    for attempt in range(max_retries):
        if len(collected) >= n_target:
            break

        still_need = n_target - len(collected)
        ask_n = still_need + 5  # ask a few extra to compensate for filter loss
        user_content = prompt_template.format(n=ask_n)
        messages = _build_messages(user_content)
        max_new_tokens = ask_n * 30

        log.info(f"  Level {level_id} ({label}): attempt {attempt+1}, asking for {ask_n}")
        try:
            if backend == "hf":
                raw = generate_hf(tokenizer, model_or_llm, messages, max_new_tokens)
            else:
                raw = generate_vllm(tokenizer, model_or_llm, SamplingParamsCls, messages, max_new_tokens)
        except Exception as e:
            log.warning(f"  Generation failed on attempt {attempt+1}: {e}")
            continue

        candidates = parse_lines(raw)
        new = sbert_filter(
            candidates, canon_emb, sbert, seen_sigs,
            SBERT_LOWER, SBERT_UPPER, TOKEN_OVERLAP_MAX, canonical,
        )
        collected.extend(new[:still_need])
        log.info(
            f"  Level {level_id}: +{len(new[:still_need])} accepted "
            f"({len(candidates)} parsed, {len(collected)}/{n_target} total)"
        )

    if len(collected) < n_target:
        log.warning(
            f"  Level {level_id} ({label}): only {len(collected)}/{n_target} "
            "instructions passed filters after all retries."
        )

    # Tag with level metadata
    for item in collected:
        item["complexity_level"] = level_id
        item["complexity_label"] = label

    return collected[:n_target]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate summarization instruction pool (simple→complex) using Qwen 8B"
    )
    parser.add_argument(
        "--backend", choices=["hf", "vllm"], default="hf",
        help="'hf' = HF Transformers (bfloat16, device_map=auto); 'vllm' = vLLM engine (H100)",
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID,
        help=f"HF model repo id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--n-per-level", type=int, default=25,
        help="Instructions to generate per complexity level (default: 25 → 100 total)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: gensens/data/summ_instruction_pool.jsonl)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="vLLM GPU memory fraction (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="vLLM max model length (default: 4096)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=4,
        help="Per-level generation retries if filter yield is low (default: 4)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "summ_instruction_pool.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Summarization Instruction Pool Generator")
    logger.info(f"  Model:       {args.model_id}")
    logger.info(f"  Backend:     {args.backend}")
    logger.info(f"  Levels:      {len(COMPLEXITY_LEVELS)}  ({args.n_per_level} each → {len(COMPLEXITY_LEVELS) * args.n_per_level} total)")
    logger.info(f"  Output:      {output_path}")
    logger.info("=" * 60)

    # Load SBERT
    logger.info(f"Loading SBERT ({SBERT_MODEL_ID})...")
    sbert = SentenceTransformer(SBERT_MODEL_ID)
    canon_emb = sbert.encode(CANONICAL_INSTRUCTION, convert_to_tensor=True)
    logger.info("  SBERT ready.")

    # Load generation model
    SamplingParamsCls = None
    if args.backend == "hf":
        tokenizer, model_or_llm = load_hf_model(args.model_id)
    else:
        tokenizer, model_or_llm, SamplingParamsCls = load_vllm_model(
            args.model_id, args.gpu_memory_utilization, args.max_model_len
        )

    # Always include the canonical instruction (level 2, moderate)
    seen_sigs: set = {first_n_words(CANONICAL_INSTRUCTION)}
    all_records: List[Dict[str, Any]] = [
        {
            "instruction":      CANONICAL_INSTRUCTION,
            "complexity_level": 2,
            "complexity_label": "moderate",
            "sbert_similarity": 1.0,
        }
    ]

    # Generate per level
    for level_id, label, prompt_template in COMPLEXITY_LEVELS:
        logger.info(f"\n── Level {level_id}: {label} ──")
        records = generate_level(
            level_id=level_id,
            label=label,
            prompt_template=prompt_template,
            n_target=args.n_per_level,
            canon_emb=canon_emb,
            sbert=sbert,
            seen_sigs=seen_sigs,
            canonical=CANONICAL_INSTRUCTION,
            backend=args.backend,
            tokenizer=tokenizer,
            model_or_llm=model_or_llm,
            SamplingParamsCls=SamplingParamsCls,
            max_retries=args.max_retries,
            logger=logger,
        )
        all_records.extend(records)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"\n✓ Saved {len(all_records)} instructions to {output_path}")

    # Summary table
    logger.info("\nLevel breakdown:")
    logger.info(f"  {'Level':<8} {'Label':<12} {'Count':>6} {'Sim mean':>10}")
    logger.info(f"  {'-'*40}")
    from collections import defaultdict
    level_groups = defaultdict(list)
    for rec in all_records:
        level_groups[rec["complexity_level"]].append(rec["sbert_similarity"])
    for lvl, sims in sorted(level_groups.items()):
        label = next(r["complexity_label"] for r in all_records if r["complexity_level"] == lvl)
        mean_sim = sum(sims) / len(sims)
        logger.info(f"  {lvl:<8} {label:<12} {len(sims):>6} {mean_sim:>10.4f}")


if __name__ == "__main__":
    main()
