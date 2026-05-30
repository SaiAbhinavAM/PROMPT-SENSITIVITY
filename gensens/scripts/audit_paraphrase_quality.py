"""
audit_paraphrase_quality.py — Flaw §6.1 paraphrase-quality audit.

Runs DeBERTa-v3-MNLI (or a configurable NLI model) over a sample of
(base_text, paraphrased_text) pairs from a GenSens JSONL dataset and reports
the fraction of pairs that achieve **bidirectional entailment** — the strict
semantic-equivalence criterion. SBERT cos-sim ≥ 0.82 is necessary but not
sufficient: adversarial paraphrases can pass cos-sim while flipping meaning.

The output is an audit report (CSV + summary stats) that goes in the paper's
appendix as a paraphrase-quality table.

Usage:
    python audit_paraphrase_quality.py \\
        --jsonl gensens/data/gensens_summarization_5inst_3var.jsonl \\
        --sample-size 100 \\
        --nli-model microsoft/deberta-v3-base-mnli \\
        --output paraphrase_audit.csv

Notes:
- NLI models output (entailment, neutral, contradiction) probabilities.
  "Bidirectional entailment" = P(entail|A→B) > τ AND P(entail|B→A) > τ.
- Default τ = 0.5 (the natural decision boundary).
- Runs on GPU if available, otherwise CPU (slow but works).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_pairs(jsonl_path: Path) -> List[Tuple[str, str, str, str]]:
    """Yield (instance_id, task, base_text, paraphrased_text) tuples."""
    pairs: List[Tuple[str, str, str, str]] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            base = rec.get("base_text", "")
            task = rec.get("task", "")
            iid = rec.get("instance_id", "")
            for v in rec.get("variants", []) or []:
                pp = v.get("paraphrased_text", "")
                if base and pp:
                    pairs.append((iid, task, base, pp))
    return pairs


def run_nli(model_name: str, pairs: List[Tuple[str, str]], batch_size: int = 16):
    """Return a list of (P(entail), P(neutral), P(contradict)) per pair.

    `pairs` is a list of (premise, hypothesis) strings.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading NLI model {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # MNLI label order varies by model; deberta-v3-mnli is (contradiction,
    # neutral, entailment). We re-order to (entail, neutral, contradict).
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    label_to_idx = {v: k for k, v in id2label.items()}
    e_idx = label_to_idx.get("entailment", 2)
    n_idx = label_to_idx.get("neutral", 1)
    c_idx = label_to_idx.get("contradiction", 0)

    out: List[Tuple[float, float, float]] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = tok(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation=True, padding=True, max_length=512, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = logits.softmax(dim=-1).cpu().numpy()
            for row in probs:
                out.append((float(row[e_idx]), float(row[n_idx]), float(row[c_idx])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="GenSens JSONL dataset (with variants[].paraphrased_text)")
    ap.add_argument("--sample-size", type=int, default=100,
                    help="Number of (base, paraphrase) pairs to audit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nli-model", type=str,
                    default="microsoft/deberta-v3-base-mnli",
                    help="HuggingFace NLI model id")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Per-direction entailment threshold for "
                         "'bidirectional entailment' (default 0.5)")
    ap.add_argument("--output", type=Path, default=Path("paraphrase_audit.csv"))
    args = ap.parse_args()

    if not args.jsonl.exists():
        logger.error(f"Input JSONL not found: {args.jsonl}")
        sys.exit(1)

    pairs = load_pairs(args.jsonl)
    if not pairs:
        logger.error("No (base, paraphrase) pairs found in the JSONL.")
        sys.exit(1)
    logger.info(f"Loaded {len(pairs)} pairs from {args.jsonl}")

    random.seed(args.seed)
    if args.sample_size < len(pairs):
        pairs = random.sample(pairs, args.sample_size)
        logger.info(f"Sampled {len(pairs)} pairs for audit (seed={args.seed})")

    # Build forward and reverse pair lists.
    fwd_inputs = [(b, p) for _, _, b, p in pairs]
    rev_inputs = [(p, b) for _, _, b, p in pairs]

    logger.info("Running NLI on forward direction (base → paraphrase) …")
    fwd = run_nli(args.nli_model, fwd_inputs)
    logger.info("Running NLI on reverse direction (paraphrase → base) …")
    rev = run_nli(args.nli_model, rev_inputs)

    # Build per-pair report.
    n_bi = 0
    n_one = 0
    n_neither = 0
    out_rows = []
    for (iid, task, base, para), (e_f, n_f, c_f), (e_r, n_r, c_r) in zip(pairs, fwd, rev):
        bi = e_f > args.tau and e_r > args.tau
        any_dir = (e_f > args.tau) or (e_r > args.tau)
        if bi:
            verdict = "bidirectional"
            n_bi += 1
        elif any_dir:
            verdict = "one_direction"
            n_one += 1
        else:
            verdict = "neither"
            n_neither += 1
        out_rows.append({
            "instance_id": iid, "task": task,
            "base_text": base, "paraphrased_text": para,
            "entail_fwd": round(e_f, 4), "neutral_fwd": round(n_f, 4),
            "contradict_fwd": round(c_f, 4),
            "entail_rev": round(e_r, 4), "neutral_rev": round(n_r, 4),
            "contradict_rev": round(c_r, 4),
            "verdict": verdict,
        })

    # Write CSV.
    import csv as _csv
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    logger.info(f"Per-pair audit written to {args.output}")

    # Summary.
    n = len(out_rows)
    print()
    print(f"=== Paraphrase quality audit (n={n}, τ={args.tau}) ===")
    print(f"  Bidirectional entailment : {n_bi:4d} ({100*n_bi/n:5.1f}%)")
    print(f"  One-direction entailment : {n_one:4d} ({100*n_one/n:5.1f}%)")
    print(f"  Neither direction        : {n_neither:4d} ({100*n_neither/n:5.1f}%)")
    print()
    print("The 'Bidirectional entailment' % is the paper-appendix table value.")


if __name__ == "__main__":
    main()
