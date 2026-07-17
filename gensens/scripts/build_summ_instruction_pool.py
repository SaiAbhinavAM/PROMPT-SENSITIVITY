#!/usr/bin/env python3
"""
build_summ_instruction_pool.py
==============================
Builds summ_instruction_pool.jsonl from 50 hand-crafted summarization
instructions spanning 10 distinct dimensions.

Each dimension tests a different axis of prompt variation — the key property
for a prompt sensitivity benchmark. All 50 instructions are:
  • Unique in wording
  • Unique in style / register
  • Unique in implied meaning / emphasis

Pipeline:
  1. Define 50 hand-crafted prompts (5 × 10 dimensions)
  2. Compute SBERT similarity to canonical (relevance gate)
  3. Run pairwise diversity check — flag any pair above MAX_PAIRWISE_SIM
  4. Assign complexity level by word count
  5. Save to gensens/data/summ_instruction_pool.jsonl

Output record fields:
  instruction      — the standalone prompt text
  dimension        — dimension name (e.g. "role_based")
  dimension_idx    — 1-10
  sbert_similarity — cosine similarity to canonical instruction
  complexity_level — 1 (simple) / 2 (moderate) / 3 (detailed) / 4 (complex)
  complexity_label — "simple" / "moderate" / "detailed" / "complex"

Usage:
  python scripts/build_summ_instruction_pool.py [--check-only]
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL = (
    "Summarize the following news article in 3-4 sentences, "
    "capturing the main events and key details."
)

SBERT_MODEL      = "all-mpnet-base-v2"
SBERT_LOWER      = 0.40   # minimum relevance to canonical (is it about summarization?)
MAX_PAIRWISE_SIM = 0.85   # flag pairs above this as too similar

LEVEL_BINS = [(10, 1), (20, 2), (35, 3)]  # (max_words, level); else 4

# ─────────────────────────────────────────────────────────────────────────────
# The 50 hand-crafted prompts — 5 per dimension
#
# Each tuple: (dimension_name, dimension_idx, instruction_text)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS: List[Tuple[str, int, str]] = [

    # ── Dimension 1: Direct Command ──────────────────────────────────────────
    # Plain imperative with no framing, role, or format constraint.
    # Vary: sentence count, verb choice, what to "capture"
    ("direct_command", 1,
     "Summarize the following news article in 3-4 sentences, capturing the main events and key details."),
    ("direct_command", 1,
     "Extract the most newsworthy information from the following news article."),
    ("direct_command", 1,
     "Recap the key developments from the following news article."),
    ("direct_command", 1,
     "Condense the following news article into its most essential points."),
    ("direct_command", 1,
     "Restate the main points of the following news article in your own words."),

    # ── Dimension 2: Question Form ────────────────────────────────────────────
    # Framed as a question the reader answers about the article.
    # Vary: what aspect is asked about
    ("question_form", 2,
     "What events or decisions led to the situation described in the following news article?"),
    ("question_form", 2,
     "What is the main story being reported in the following news article?"),
    ("question_form", 2,
     "What happened according to the following news article?"),
    ("question_form", 2,
     "What is the most significant or surprising development described in the following news article?"),
    ("question_form", 2,
     "What does the following news article want the reader to understand?"),

    # ── Dimension 3: Role-Based ───────────────────────────────────────────────
    # Assigns a professional persona before asking for the summary.
    # Vary: profession, purpose
    ("role_based", 3,
     "As a news editor, write a short summary of the following article for the front page."),
    ("role_based", 3,
     "You are a student writing a research report. Summarize the following news article to use as a reference source."),
    ("role_based", 3,
     "Acting as a research assistant, extract the key findings from the following news article."),
    ("role_based", 3,
     "Imagine you are a news anchor introducing this story on television. Write a brief introduction based on the following article."),
    ("role_based", 3,
     "As a librarian creating a catalog entry, write a short description of what the following news article covers."),

    # ── Dimension 4: Constraint-Based ────────────────────────────────────────
    # Imposes a hard output constraint — vary the CONSTRAINT TYPE, not just number.
    # Types: word-count cap, rigid 3-part template, vocabulary restriction,
    #        single-sentence, structured label format
    ("constraint_based", 4,
     "In no more than 50 words, summarize the following news article."),
    ("constraint_based", 4,
     "Give a three-part summary of the following news article: context, main event, and outcome."),
    ("constraint_based", 4,
     "Write a one-sentence summary of the following news article."),
    ("constraint_based", 4,
     "Summarize the following news article using only plain, everyday words — avoid jargon or technical language."),
    ("constraint_based", 4,
     "Summarize the following news article as a series of short, telegraphic phrases — no full sentences needed."),

    # ── Dimension 5: Analytical Framing ──────────────────────────────────────
    # Asks the model to analyse structure rather than just re-state facts.
    # Vary: analytical lens (5W, cause-effect, claim-evidence, conflict)
    ("analytical_framing", 5,
     "For the following news article, identify the who, what, when, where, and why."),
    ("analytical_framing", 5,
     "What is the central claim made in the following news article and what evidence is provided to support it?"),
    ("analytical_framing", 5,
     "Analyse the following news article and explain the cause-and-effect relationship it describes."),
    ("analytical_framing", 5,
     "Break down the following news article into its main claim and the key facts that support it."),
    ("analytical_framing", 5,
     "What is the underlying conflict or issue described in the following news article, and how is it being addressed?"),

    # ── Dimension 6: Output Format ────────────────────────────────────────────
    # Specifies a particular non-prose structure for the output.
    # Vary: bullet points, numbered list, headline+body, tweet, structured template
    ("output_format", 6,
     "Write 3 bullet points summarizing the most important information from the following news article."),
    ("output_format", 6,
     "Create a short headline and a two-sentence summary for the following news article."),
    ("output_format", 6,
     "Summarize the following news article as if writing a tweet — keep it clear, punchy, and under 280 characters."),
    ("output_format", 6,
     "Write a numbered list of the 3 most important takeaways from the following news article."),
    ("output_format", 6,
     "Summarize the following news article using this format: Main Event | Key People | Outcome."),

    # ── Dimension 7: Audience / Perspective ──────────────────────────────────
    # Tailors the summary for a specific reader type or reading context.
    # Vary: audience (child, expert, executive, non-native speaker, general public)
    ("audience_perspective", 7,
     "Explain the following news article to someone who has no background knowledge on the topic."),
    ("audience_perspective", 7,
     "Summarize the following news article for a busy executive who only has 30 seconds to read it."),
    ("audience_perspective", 7,
     "Write a summary of the following news article that would be easy for a high school student to understand."),
    ("audience_perspective", 7,
     "Summarize the following news article for a non-native English speaker — use simple, clear language."),
    ("audience_perspective", 7,
     "Summarize the following news article in a way that a domain expert would find concise and professionally useful."),

    # ── Dimension 8: Emphasis / Focus ────────────────────────────────────────
    # Directs the model to attend to a specific aspect of the content.
    # Vary: people, problem/solution, numbers, timeline, impact
    ("emphasis_focus", 8,
     "Summarize the following news article focusing specifically on the people involved and their roles."),
    ("emphasis_focus", 8,
     "What problem is described in the following news article, and what solutions or responses are mentioned?"),
    ("emphasis_focus", 8,
     "Summarize the following news article paying special attention to any numbers, statistics, or specific facts mentioned."),
    ("emphasis_focus", 8,
     "Describe the timeline of events in the following news article in chronological order."),
    ("emphasis_focus", 8,
     "Summarize the following news article by highlighting its potential impact or consequences."),

    # ── Dimension 9: Tone / Register ──────────────────────────────────────────
    # Specifies the voice or register of the summary output.
    # Vary: formal, casual, dry-factual, professional briefing, engaging
    ("tone_register", 9,
     "Write a formal, objective summary of the following news article."),
    ("tone_register", 9,
     "Give a casual, conversational overview of what the following news article is about."),
    ("tone_register", 9,
     "Write a dry, factual account of the events described in the following news article."),
    ("tone_register", 9,
     "Write a vivid, journalistic account of the events described in the following news article."),
    ("tone_register", 9,
     "Write an engaging, reader-friendly summary of the following news article that makes someone want to read the full story."),

    # ── Dimension 10: Completeness Framing ───────────────────────────────────
    # Emphasises what to include, exclude, or prioritise in the summary.
    # Vary: exhaustive, single-takeaway, context-heavy, minimal, full-picture
    ("completeness_framing", 10,
     "Summarize the following news article capturing both the factual content and why the story matters."),
    ("completeness_framing", 10,
     "What is the single most important takeaway from the following news article?"),
    ("completeness_framing", 10,
     "Summarize the following news article from start to finish, preserving the logical flow of events."),
    ("completeness_framing", 10,
     "Give a minimal summary of the following news article — just the essential facts, nothing extra."),
    ("completeness_framing", 10,
     "Summarize the following news article so that someone who did not read it would have a complete understanding of the story."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

LABELS = {1: "simple", 2: "moderate", 3: "detailed", 4: "complex"}


def assign_level(text: str) -> int:
    n = len(text.split())
    for max_w, lvl in LEVEL_BINS:
        if n <= max_w:
            return lvl
    return 4


def build_records(
    sbert: SentenceTransformer,
    canon_emb,
) -> List[Dict[str, Any]]:
    records = []
    for dim_name, dim_idx, instr in PROMPTS:
        emb = sbert.encode(instr, convert_to_tensor=True)
        sim = float(st_util.cos_sim(canon_emb, emb).item())
        lvl = assign_level(instr)
        records.append({
            "instruction":      instr,
            "dimension":        dim_name,
            "dimension_idx":    dim_idx,
            "sbert_similarity": round(sim, 4),
            "complexity_level": lvl,
            "complexity_label": LABELS[lvl],
        })
    return records


def check_pairwise_diversity(
    records: List[Dict[str, Any]],
    sbert: SentenceTransformer,
    threshold: float,
) -> List[Tuple[float, str, str]]:
    """Return list of (sim, instr_a, instr_b) pairs that exceed threshold."""
    instrs = [r["instruction"] for r in records]
    embs   = sbert.encode(instrs, convert_to_tensor=False, show_progress_bar=False)
    sim_mat = st_util.cos_sim(embs, embs).numpy()
    n = len(instrs)
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_mat[i][j])
            if s > threshold:
                violations.append((s, instrs[i], instrs[j]))
    violations.sort(reverse=True)
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--check-only", action="store_true",
                        help="Run diversity check and print stats without saving")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger(__name__)

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "data" / "summ_instruction_pool.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Summarization Instruction Pool Builder")
    log.info(f"  Prompts:  {len(PROMPTS)} hand-crafted ({len(PROMPTS)//5} dimensions × 5)")
    log.info(f"  Output:   {out_path}")
    log.info("=" * 60)

    assert len(PROMPTS) == 50, f"Expected 50 prompts, got {len(PROMPTS)}"

    # Load SBERT
    log.info(f"Loading SBERT ({SBERT_MODEL})...")
    sbert     = SentenceTransformer(SBERT_MODEL)
    canon_emb = sbert.encode(CANONICAL, convert_to_tensor=True)
    log.info("  SBERT ready.")

    # Build records
    records = build_records(sbert, canon_emb)

    # ── Relevance check ───────────────────────────────────────────────────────
    low_rel = [r for r in records if r["sbert_similarity"] < SBERT_LOWER]
    if low_rel:
        log.warning(f"  {len(low_rel)} instructions below relevance threshold ({SBERT_LOWER}):")
        for r in low_rel:
            log.warning(f"    [{r['sbert_similarity']:.3f}] {r['instruction']}")

    # ── Pairwise diversity check ──────────────────────────────────────────────
    log.info(f"Running pairwise diversity check (threshold={MAX_PAIRWISE_SIM})...")
    violations = check_pairwise_diversity(records, sbert, MAX_PAIRWISE_SIM)
    if violations:
        log.warning(f"  {len(violations)} pairs exceed similarity threshold {MAX_PAIRWISE_SIM}:")
        for s, a, b in violations[:10]:
            log.warning(f"    [{s:.3f}]")
            log.warning(f"      A: {a}")
            log.warning(f"      B: {b}")
    else:
        log.info(f"  All pairs are below {MAX_PAIRWISE_SIM} — pool is diverse.")

    # ── Pairwise stats ────────────────────────────────────────────────────────
    instrs  = [r["instruction"] for r in records]
    embs    = sbert.encode(instrs, convert_to_tensor=False, show_progress_bar=False)
    sim_mat = st_util.cos_sim(embs, embs).numpy()
    n = len(instrs)
    all_sims = [sim_mat[i][j] for i in range(n) for j in range(i+1, n)]
    arr = np.array(all_sims)
    log.info(f"\nPairwise similarity across {len(all_sims)} pairs:")
    log.info(f"  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
             f"max={arr.max():.3f}  min={arr.min():.3f}")

    if args.check_only:
        log.info("\n--check-only: not saving.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    log.info(f"\n✓ Saved {len(records)} instructions → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    groups: Dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["dimension"]].append(r)

    log.info(f"\n{'Dim':>3}  {'Dimension':<22}  {'N':>3}  {'SimMean':>8}  {'SimMin':>8}")
    log.info("-" * 55)
    for dim_name, dim_idx, _ in PROMPTS[::5]:   # one header per dimension
        recs = groups[dim_name]
        sims = [r["sbert_similarity"] for r in recs]
        log.info(f"  {dim_idx:>2}  {dim_name:<22}  {len(recs):>3}  "
                 f"{sum(sims)/len(sims):>8.4f}  {min(sims):>8.4f}")

    # ── Level breakdown ───────────────────────────────────────────────────────
    lvl_groups: Dict[int, list] = defaultdict(list)
    for r in records:
        lvl_groups[r["complexity_level"]].append(r)
    log.info(f"\n{'Lvl':<5} {'Label':<14} {'Count':>6}")
    log.info("-" * 28)
    for lvl in sorted(lvl_groups):
        log.info(f"  {lvl:<4} {LABELS[lvl]:<14} {len(lvl_groups[lvl]):>6}")
    log.info(f"  {'TOT':<4} {'':<14} {len(records):>6}")


if __name__ == "__main__":
    main()
