#!/usr/bin/env python3
"""
compare_models.py — cross-model replication analysis for the crossed-design
benchmark. Given two completed runs (same articles+seeds, different generation
model), quantify whether the SENSITIVITY findings replicate across models.

Pure CPU (numpy/scipy). Reads cell_metrics_scored.jsonl from each run dir.

The headline replication number is the **dimension-ranking Spearman** between the
two models: high (>0.7) => the same instruction dimensions are sensitive in both
models => the finding generalizes beyond a single model. Because the Sensitivity
composite is rank-normalized WITHIN each run, only the ORDER is comparable across
runs (not absolute values) — so all cross-run comparisons here are rank-based.

Also reports:
  - seed-level ranking Spearman (50 prompts — a finer replication signal),
  - the same comparison on the absolute anchor `sms_drift` (run-independent),
  - top-k / bottom-k set overlap,
  - per-pool Sensitivity in each model,
  - a side-by-side per-dimension table.

Usage:
  python compare_models.py \
      --run_a results_30article_bestscorers/results --label_a Llama-3.1-8B \
      --run_b results_qwen/results               --label_b Qwen2.5-7B
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compare_models")


def _load(run_dir):
    cells = common.read_jsonl(Path(run_dir) / "cell_metrics_scored.jsonl")
    if not cells:
        raise FileNotFoundError(f"No cell_metrics_scored.jsonl in {run_dir}")
    return cells


def _group_mean(cells, group_key, value_key):
    d = defaultdict(list)
    for c in cells:
        d[c[group_key]].append(c[value_key])
    return {k: float(np.mean(v)) for k, v in d.items()}


def _rank_corr(map_a, map_b):
    """Spearman + Pearson over the keys present in BOTH maps."""
    keys = sorted(set(map_a) & set(map_b))
    if len(keys) < 3:
        return {"n": len(keys), "spearman": None, "pearson": None, "keys": keys}
    a = [map_a[k] for k in keys]
    b = [map_b[k] for k in keys]
    sr = spearmanr(a, b)
    return {
        "n": len(keys),
        "spearman": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "pearson": float(pearsonr(a, b)[0]),
        "keys": keys,
    }


def _set_overlap(map_a, map_b, k, top=True):
    def pick(m):
        order = sorted(m, key=m.get, reverse=top)
        return set(order[:k])
    sa, sb = pick(map_a), pick(map_b)
    inter = sa & sb
    union = sa | sb
    return {"k": k, "top": top, "a": sorted(sa), "b": sorted(sb),
            "shared": sorted(inter), "jaccard": len(inter) / len(union) if union else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--label_a", default="model_A")
    ap.add_argument("--label_b", default="model_B")
    ap.add_argument("--out", default=None, help="Output JSON path (default: alongside run_b).")
    args = ap.parse_args()

    A, B = _load(args.run_a), _load(args.run_b)
    log.info(f"{args.label_a}: {len(A)} cells | {args.label_b}: {len(B)} cells")

    # ── Dimension-level (headline) ────────────────────────────────────────
    dimA = _group_mean(A, "dimension", "sensitivity")
    dimB = _group_mean(B, "dimension", "sensitivity")
    dim_corr = _rank_corr(dimA, dimB)

    # Absolute-anchor cross-check (run-independent sms_drift)
    driftA = _group_mean(A, "dimension", "sms_drift")
    driftB = _group_mean(B, "dimension", "sms_drift")
    drift_corr = _rank_corr(driftA, driftB)

    # ── Seed-level (finer: do the same 50 prompts rank alike?) ────────────
    seedA = _group_mean(A, "seed_id", "sensitivity")
    seedB = _group_mean(B, "seed_id", "sensitivity")
    seed_corr = _rank_corr(seedA, seedB)

    # ── Pool-level Sensitivity per model ──────────────────────────────────
    poolA = _group_mean(A, "pool", "sensitivity")
    poolB = _group_mean(B, "pool", "sensitivity")

    n_dims = dim_corr["n"]
    result = {
        "label_a": args.label_a, "label_b": args.label_b,
        "n_cells_a": len(A), "n_cells_b": len(B),
        "dimension_ranking": {k: v for k, v in dim_corr.items() if k != "keys"},
        "dimension_ranking_on_sms_drift_anchor": {k: v for k, v in drift_corr.items() if k != "keys"},
        "seed_ranking": {k: v for k, v in seed_corr.items() if k != "keys"},
        "top3_overlap": _set_overlap(dimA, dimB, 3, top=True),
        "bottom3_overlap": _set_overlap(dimA, dimB, 3, top=False),
        "pool_sensitivity": {args.label_a: poolA, args.label_b: poolB},
        "verdict": (
            "REPLICATES — same dimensions sensitive in both models"
            if dim_corr["spearman"] is not None and dim_corr["spearman"] > 0.7 else
            "does NOT cleanly replicate — inspect divergences"
        ),
    }

    out_path = Path(args.out) if args.out else Path(args.run_b) / f"compare_{args.label_a}_vs_{args.label_b}.json"
    out_path.write_text(json.dumps(result, indent=2))

    # ── Pretty print ──────────────────────────────────────────────────────
    print(f"\n{'='*72}\nCROSS-MODEL REPLICATION: {args.label_a}  vs  {args.label_b}\n{'='*72}")
    print(f"Dimension-ranking Spearman = {dim_corr['spearman']:+.3f} (p={dim_corr.get('spearman_p', float('nan')):.4g}, n={n_dims} dims)")
    print(f"  cross-check on sms_drift  = {drift_corr['spearman']:+.3f}")
    print(f"Seed-ranking Spearman      = {seed_corr['spearman']:+.3f} (p={seed_corr.get('spearman_p', float('nan')):.4g}, n={seed_corr['n']} seeds)")
    print(f"Top-3 shared:    {result['top3_overlap']['shared']}  (Jaccard {result['top3_overlap']['jaccard']:.2f})")
    print(f"Bottom-3 shared: {result['bottom3_overlap']['shared']}  (Jaccard {result['bottom3_overlap']['jaccard']:.2f})")
    print(f"\n>>> {result['verdict']}\n")

    # side-by-side dimension table (sorted by model A rank)
    keys = sorted(set(dimA) & set(dimB), key=lambda d: dimA[d], reverse=True)
    ra = {d: i + 1 for i, d in enumerate(sorted(dimA, key=dimA.get, reverse=True))}
    rb = {d: i + 1 for i, d in enumerate(sorted(dimB, key=dimB.get, reverse=True))}
    print(f"{'dimension':24s} {args.label_a[:12]:>12s} {'rank':>5s} | {args.label_b[:12]:>12s} {'rank':>5s} | {'Δrank':>5s}")
    for d in keys:
        print(f"{d:24s} {dimA[d]:>12.3f} {ra[d]:>5d} | {dimB[d]:>12.3f} {rb[d]:>5d} | {rb[d]-ra[d]:>+5d}")

    print(f"\n[compare_models.py] wrote {out_path}")


if __name__ == "__main__":
    main()
