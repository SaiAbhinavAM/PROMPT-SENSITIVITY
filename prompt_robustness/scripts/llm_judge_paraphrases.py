"""
llm_judge_paraphrases.py — GPT-4o-mini-as-judge for paraphrase quality.

PUBLICATION_SPEC §8.1 (Validation Prong 1): in lieu of human annotation, we
use an LLM-as-judge protocol (Zheng et al., NeurIPS 2023; Liu et al., EMNLP
2023) to rate paraphrase quality on a 1–5 Likert scale.

Two modes:

  --mode rate ............ Rate N stratified (base, paraphrase) pairs from a
                           GenSens JSONL. Outputs:
                             paraphrase_judge_ratings.jsonl  (per-pair)
                             paraphrase_judge_summary.json   (aggregates)

  --mode calibrate ....... Run on RobustAlpacaEval's human-verified
                           paraphrases (or any (base, paraphrase, human_score)
                           CSV) and report Cohen's κ + Spearman ρ between
                           GPT-4o-mini ratings and the human gold labels.
                           This is the LLM-as-judge legitimacy claim.

USAGE
-----
  # Rate 500 sampled pairs from a GenSens dataset
  export OPENAI_API_KEY=sk-...
  python prompt_robustness/scripts/llm_judge_paraphrases.py \\
      --mode rate \\
      --input gensens/data/gensens_summarization_200inst_8var.jsonl \\
      --output-dir results/judge_ratings \\
      --n-samples 500 \\
      --stratify-by-strategy

  # Calibrate against RobustAlpacaEval human ratings
  python prompt_robustness/scripts/llm_judge_paraphrases.py \\
      --mode calibrate \\
      --human-ratings robustalpaca_human_ratings.csv \\
      --output-dir results/judge_calibration

OUTPUT
------
For --mode rate:
  paraphrase_judge_ratings.jsonl   {pair_id, base, paraphrase, judge_rating,
                                     raw_response, model, latency_s, task,
                                     strategy, strategy_family}
  paraphrase_judge_summary.json    {n_rated, mean, std, percentiles,
                                     per_task_mean, per_strategy_mean,
                                     per_family_mean, low_quality_rate (≤2)}

For --mode calibrate:
  paraphrase_judge_calibration.json {cohen_kappa, spearman_rho,
                                      n_pairs, confusion_matrix, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("llm_judge_paraphrases")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt — kept ASCII to avoid any API encoding quirks.
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether two sentences are paraphrases of each other. "
    "A 5 means perfect paraphrase: identical meaning, different wording. "
    "A 4 means good paraphrase: very minor nuance shift. "
    "A 3 means acceptable paraphrase: some intent drift. "
    "A 2 means poor paraphrase: meaningful intent change. "
    "A 1 means not a paraphrase: different meaning. "
    "Output ONLY the integer rating. Do not explain."
)

JUDGE_USER_TEMPLATE = (
    "Base:       {base}\n"
    "Paraphrase: {candidate}\n\n"
    "Rating (1-5):"
)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_OPENAI_CLIENT = None


def _get_openai_client():
    """Lazy-import the OpenAI client. Fails loudly if not installed/keyed."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("ERROR: pip install openai>=1.0  (required for --mode rate/calibrate)")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY environment variable is not set.")
    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


_RATING_PATTERN = re.compile(r"\b([1-5])\b")


def _judge_one_pair(
    client,
    base: str,
    candidate: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> Tuple[Optional[int], str, float]:
    """Rate one (base, paraphrase) pair on the 1–5 Likert scale.

    Returns ``(rating, raw_response_text, latency_seconds)``.
    `rating=None` indicates a parse failure (the raw response is logged).
    """
    user_msg = JUDGE_USER_TEMPLATE.format(base=base, candidate=candidate)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,  # deterministic rating
                max_tokens=4,
                seed=42,
            )
            latency = time.time() - t0
            raw = (resp.choices[0].message.content or "").strip()
            m = _RATING_PATTERN.search(raw)
            rating = int(m.group(1)) if m else None
            return rating, raw, latency
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            logger.warning(f"OpenAI call failed (attempt {attempt+1}): {e}. retrying in {wait}s")
            time.sleep(wait)
    logger.error(f"Giving up on pair after retries: {last_err}")
    return None, f"ERROR: {last_err}", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Loaders / stratified sampling
# ─────────────────────────────────────────────────────────────────────────────

def _iter_gensens_pairs(jsonl_paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Flatten one or more GenSens JSONL files into (base, paraphrase) pairs.

    Each yielded dict carries: pair_id, instance_id, task, base, candidate,
    strategy, strategy_family (when present in the source record).
    """
    pairs: List[Dict[str, Any]] = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                base = rec.get("base_text", "")
                task = rec.get("task", "")
                instance_id = rec.get("instance_id", "")
                for v in rec.get("variants", []) or []:
                    pid = f"{instance_id}::var{v.get('variant_idx', '?')}"
                    pairs.append({
                        "pair_id":          pid,
                        "instance_id":      instance_id,
                        "task":             task,
                        "base":             base,
                        "candidate":        v.get("paraphrased_text", ""),
                        "strategy":         v.get("strategy", "unknown"),
                        "strategy_family":  v.get("strategy_family", "unknown"),
                    })
    return pairs


def _stratified_sample(
    pairs: List[Dict[str, Any]],
    n_samples: int,
    seed: int = 42,
    by: str = "strategy",
) -> List[Dict[str, Any]]:
    """Sample ~equal numbers per stratum, then back-fill randomly to n.

    `by`: 'strategy' | 'strategy_family' | 'task' — the stratification key.
    """
    rng = random.Random(seed)
    if n_samples >= len(pairs):
        return list(pairs)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        buckets[p.get(by, "unknown")].append(p)
    strata = sorted(buckets.keys())
    per_stratum = max(1, n_samples // max(1, len(strata)))
    sampled: List[Dict[str, Any]] = []
    for s in strata:
        rng.shuffle(buckets[s])
        sampled.extend(buckets[s][:per_stratum])
    # Back-fill if rounding leaves us short.
    remaining = [p for p in pairs if p not in sampled]
    rng.shuffle(remaining)
    while len(sampled) < n_samples and remaining:
        sampled.append(remaining.pop())
    return sampled[:n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# --mode rate
# ─────────────────────────────────────────────────────────────────────────────

def run_rate_mode(args) -> int:
    inputs = [args.input] if args.input else args.inputs or []
    if not inputs:
        sys.exit("ERROR: --input or --inputs is required for --mode rate")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path  = out_dir / "paraphrase_judge_ratings.jsonl"
    summary_path  = out_dir / "paraphrase_judge_summary.json"

    print(f"→ Loading pairs from {len(inputs)} JSONL file(s) …")
    pairs = _iter_gensens_pairs(inputs)
    print(f"  loaded {len(pairs)} (base, paraphrase) pairs")

    # Resume support — skip pair_ids already rated.
    done_ids: set = set()
    if ratings_path.exists():
        with open(ratings_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["pair_id"])
                except Exception:
                    pass
        if done_ids:
            print(f"↻ Resuming: {len(done_ids)} pairs already rated, will skip")

    pending = [p for p in pairs if p["pair_id"] not in done_ids]
    if args.stratify_by_strategy:
        sampled = _stratified_sample(pending, args.n_samples, args.seed, by="strategy")
    elif args.stratify_by_task:
        sampled = _stratified_sample(pending, args.n_samples, args.seed, by="task")
    else:
        rng = random.Random(args.seed)
        rng.shuffle(pending)
        sampled = pending[:args.n_samples]
    print(f"  sampling {len(sampled)} pairs (target n={args.n_samples})")

    client = _get_openai_client()
    rated: List[Dict[str, Any]] = []
    fout = open(ratings_path, "a", encoding="utf-8")
    try:
        for i, pair in enumerate(sampled, 1):
            rating, raw, latency = _judge_one_pair(
                client, pair["base"], pair["candidate"],
                model=args.model, max_retries=args.max_retries,
            )
            record = dict(pair)
            record.update({
                "judge_rating": rating,
                "raw_response": raw,
                "latency_s":    round(latency, 3),
                "model":        args.model,
            })
            rated.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(sampled):
                avg = mean([r["judge_rating"] for r in rated
                            if r["judge_rating"] is not None]) if rated else 0.0
                print(f"  [{i}/{len(sampled)}] running mean = {avg:.2f}")
    finally:
        fout.close()

    # Aggregate summary (all rated rows on disk, including prior runs).
    all_rated: List[Dict[str, Any]] = []
    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_rated.append(json.loads(line))
            except Exception:
                pass
    summary = _summarise_ratings(all_rated)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ wrote {ratings_path}")
    print(f"✓ wrote {summary_path}")
    print(f"\nMean Likert rating: {summary['mean_rating']:.3f}  "
          f"(n={summary['n_rated']}, low-quality rate ≤2: "
          f"{100 * summary['low_quality_rate']:.1f}%)")
    return 0


def _summarise_ratings(rated: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rated if isinstance(r.get("judge_rating"), int)]
    if not valid:
        return {"n_rated": 0, "n_parsed_failures": len(rated)}

    ratings = [r["judge_rating"] for r in valid]
    summary = {
        "n_rated":            len(valid),
        "n_parse_failures":   len(rated) - len(valid),
        "mean_rating":        float(mean(ratings)),
        "median_rating":      float(median(ratings)),
        "std_rating":         float(stdev(ratings)) if len(ratings) > 1 else 0.0,
        "low_quality_rate":   float(sum(1 for x in ratings if x <= 2) / len(ratings)),
        "high_quality_rate":  float(sum(1 for x in ratings if x >= 4) / len(ratings)),
        "rating_distribution": dict(Counter(ratings)),
    }

    def _group_mean(key: str) -> Dict[str, float]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for r in valid:
            groups[str(r.get(key, "unknown"))].append(r["judge_rating"])
        return {k: float(mean(v)) for k, v in groups.items()}

    summary["per_task_mean"]      = _group_mean("task")
    summary["per_strategy_mean"]  = _group_mean("strategy")
    summary["per_family_mean"]    = _group_mean("strategy_family")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# --mode calibrate
# ─────────────────────────────────────────────────────────────────────────────

def run_calibrate_mode(args) -> int:
    """Run GPT-4o-mini on a human-labelled set and report agreement metrics.

    Input CSV columns (any of): base, paraphrase, human_score
    Aliases accepted: prompt/original/source for base;
    candidate/paraphrased/variant for paraphrase; rating/label/score for human.
    """
    if not args.human_ratings or not Path(args.human_ratings).exists():
        sys.exit("ERROR: --human-ratings CSV path is required and must exist")

    print(f"→ Loading human ratings from {args.human_ratings}")
    with open(args.human_ratings, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  loaded {len(rows)} human-rated pairs")

    aliases = {
        "base":       ["base", "prompt", "original", "source"],
        "paraphrase": ["paraphrase", "candidate", "paraphrased", "variant"],
        "human":      ["human_score", "rating", "label", "score", "human"],
    }

    def resolve(row, key):
        for k in aliases[key]:
            if k in row and row[k] != "":
                return row[k]
        return None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path     = out_dir / "judge_calibration_ratings.jsonl"
    calibration_path = out_dir / "paraphrase_judge_calibration.json"

    # Cap calibration set size — keep cost predictable.
    if args.n_samples and len(rows) > args.n_samples:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n_samples]
        print(f"  capped to {len(rows)} pairs for calibration")

    client = _get_openai_client()
    rated: List[Dict[str, Any]] = []
    with open(ratings_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, 1):
            base  = resolve(row, "base")
            cand  = resolve(row, "paraphrase")
            human = resolve(row, "human")
            if base is None or cand is None or human is None:
                continue
            try:
                human_f = float(human)
            except ValueError:
                continue
            rating, raw, latency = _judge_one_pair(
                client, base, cand,
                model=args.model, max_retries=args.max_retries,
            )
            rec = {
                "pair_idx":     i,
                "base":         base, "paraphrase": cand,
                "human_score":  human_f,
                "judge_rating": rating, "raw_response": raw,
                "latency_s":    round(latency, 3),
            }
            rated.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] rated")

    metrics = _calibration_metrics(rated)
    with open(calibration_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ wrote {ratings_path}")
    print(f"✓ wrote {calibration_path}")
    print(f"\nCalibration metrics (LLM-as-judge vs human ratings):")
    print(f"  Cohen's κ (linear, integer-binned): {metrics.get('cohen_kappa', 'n/a')}")
    print(f"  Spearman ρ:                         {metrics.get('spearman_rho', 'n/a')}")
    print(f"  Mean absolute disagreement:         {metrics.get('mean_abs_disagreement', 'n/a')}")
    if metrics.get("cohen_kappa") is not None and metrics["cohen_kappa"] >= 0.60:
        print("  → ≥0.60 — supports use as a human-annotation substitute.")
    return 0


def _calibration_metrics(rated: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rated
             if isinstance(r.get("judge_rating"), int)
             and isinstance(r.get("human_score"), (int, float))]
    if len(valid) < 5:
        return {"n_pairs": len(valid), "note": "too few valid pairs"}

    judge_ratings = [r["judge_rating"] for r in valid]
    # Human scores might be 1–5 Likert or [0,1] — keep as-is for ρ, bin for κ.
    human_scores  = [float(r["human_score"]) for r in valid]

    # Spearman ρ (no SciPy dependency — manual rank correlation).
    spearman = _spearman(judge_ratings, human_scores)

    # Cohen's κ on integer-rounded human scores ∈ {1,2,3,4,5}.
    def _bin(x: float) -> int:
        if x <= 1.5: return 1
        if x <= 2.5: return 2
        if x <= 3.5: return 3
        if x <= 4.5: return 4
        return 5
    binned_human = [_bin(h) for h in human_scores]
    kappa = _cohen_kappa(judge_ratings, binned_human, n_categories=5)

    abs_diff = [abs(j - h) for j, h in zip(judge_ratings, binned_human)]
    mad = float(mean(abs_diff)) if abs_diff else None

    confusion: Dict[str, int] = defaultdict(int)
    for j, h in zip(judge_ratings, binned_human):
        confusion[f"{h}→{j}"] += 1

    return {
        "n_pairs":               len(valid),
        "cohen_kappa":           kappa,
        "spearman_rho":          spearman,
        "mean_abs_disagreement": mad,
        "exact_agreement_rate":  float(sum(1 for j, h in zip(judge_ratings, binned_human) if j == h) / len(valid)),
        "confusion_matrix":      dict(confusion),
    }


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    """Manual Spearman rank correlation (no SciPy dep)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n))
    db = sum((rb[i] - mb) ** 2 for i in range(n))
    if da == 0 or db == 0:
        return None
    return num / ((da * db) ** 0.5)


def _cohen_kappa(a: List[int], b: List[int], n_categories: int) -> Optional[float]:
    """Cohen's κ for integer ratings on a finite category set."""
    if len(a) != len(b) or len(a) == 0:
        return None
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    counts_a = Counter(a); counts_b = Counter(b)
    expected = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n)
                   for c in range(1, n_categories + 1))
    if expected >= 1.0:
        return 1.0
    return (obs - expected) / (1.0 - expected)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["rate", "calibrate"],
                   help="rate=score GenSens paraphrases; calibrate=measure agreement with human gold")
    p.add_argument("--input", help="One GenSens JSONL (for --mode rate)")
    p.add_argument("--inputs", nargs="+",
                   help="Multiple GenSens JSONLs to pool (for --mode rate)")
    p.add_argument("--human-ratings",
                   help="CSV with (base, paraphrase, human_score) columns (for --mode calibrate)")
    p.add_argument("--output-dir", required=True, help="Directory for output JSONL/JSON")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Target sample size (default 500)")
    p.add_argument("--stratify-by-strategy", action="store_true",
                   help="Stratify sampling by paraphrase strategy")
    p.add_argument("--stratify-by-task", action="store_true",
                   help="Stratify sampling by task")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model id (default gpt-4o-mini)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.mode == "rate":
        return run_rate_mode(args)
    if args.mode == "calibrate":
        return run_calibrate_mode(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
