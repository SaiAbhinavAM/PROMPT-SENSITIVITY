#!/usr/bin/env python3
"""
diagnosis_matrix.py — 2x2 diagnosis on the two INDEPENDENT axes:
  Quality axis      = CS_mean (mean correctness across the 6 variants)
  Sensitivity axis   = Sensitivity composite (spread across the 6 variants)

Thresholds are the empirical MEDIAN of each axis across all cells (so both
bins are populated by construction, modulo ties). Classification:

  High quality + Low sensitivity  -> "True Robustness"
  High quality + High sensitivity -> "Fragile"
  Low quality  + Low sensitivity  -> "Consistently Poor"
  Low quality  + High sensitivity -> "Unreliable"
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("diagnosis_matrix")


def classify(cs_mean, sensitivity, cs_median, sens_median):
    high_quality = cs_mean >= cs_median
    high_sensitivity = sensitivity >= sens_median
    if high_quality and not high_sensitivity:
        return "True Robustness"
    if high_quality and high_sensitivity:
        return "Fragile"
    if not high_quality and not high_sensitivity:
        return "Consistently Poor"
    return "Unreliable"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT
    cells = common.read_jsonl(results_dir / "cell_metrics_scored.jsonl")
    if not cells:
        raise FileNotFoundError(f"No scored cell metrics found in {results_dir}. Run aggregate_scores.py first.")

    cs_means = [c["cs_mean"] for c in cells]
    sens_vals = [c["sensitivity"] for c in cells]
    cs_median = float(np.median(cs_means))
    sens_median = float(np.median(sens_vals))
    # The thresholds are RELATIVE (this model's own medians), so the 2x2 always
    # populates both bins and "high quality" means "above THIS model's median",
    # NOT "good in absolute terms". A cell labeled "True Robustness" is
    # stable-and-above-median, which can still be mediocre. To prevent the
    # labels from being read as absolute, we surface the absolute CS_mean of
    # each quadrant below (method.md 2026-07-06 fix).
    log.info(f"RELATIVE thresholds — Quality (CS_mean) median: {cs_median:.4f} | Sensitivity median: {sens_median:.4f}")

    rows = []
    for c in cells:
        diagnosis = classify(c["cs_mean"], c["sensitivity"], cs_median, sens_median)
        rows.append({
            "article_id": c["article_id"],
            "seed_id": c["seed_id"],
            "pool": c["pool"],
            "dimension": c["dimension"],
            "cs_mean": c["cs_mean"],
            "sensitivity": c["sensitivity"],
            "diagnosis": diagnosis,
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "diagnosis.csv", index=False)
    log.info(f"Wrote {len(df)} rows to diagnosis.csv")

    counts = df["diagnosis"].value_counts()
    # Median splits populate both bins by construction, so "all 4 fire" is not
    # evidence of anything — only flag the degenerate tie case.
    empty_cells = [k for k in ["True Robustness", "Fragile", "Consistently Poor", "Unreliable"] if counts.get(k, 0) == 0]
    if empty_cells:
        log.warning(f"Diagnosis cells with ZERO members (median tie-breaking degenerate): {empty_cells}")

    order = ["True Robustness", "Fragile", "Consistently Poor", "Unreliable"]
    print("\n=== Diagnosis 4-cell distribution (RELATIVE to this model's medians) ===")
    print(f"  {'quadrant':20s} {'n':>5s}  {'abs CS_mean (mean [min,max])':>34s}  {'abs Sensitivity mean':>20s}")
    for k in order:
        sub = df[df["diagnosis"] == k]
        if len(sub):
            cs_stat = f"{sub['cs_mean'].mean():.3f} [{sub['cs_mean'].min():.3f},{sub['cs_mean'].max():.3f}]"
            s_stat = f"{sub['sensitivity'].mean():.3f}"
        else:
            cs_stat, s_stat = "-", "-"
        print(f"  {k:20s} {counts.get(k, 0):>5d}  {cs_stat:>34s}  {s_stat:>20s}")
    print(f"  NOTE: 'high quality' = CS_mean >= {cs_median:.3f} (this model's median), an abs-mediocre bar; "
          f"read the absolute CS_mean column, not the label, for real quality.")

    # Persist the per-quadrant absolute-quality summary alongside the per-cell csv.
    quad_rows = []
    for k in order:
        sub = df[df["diagnosis"] == k]
        quad_rows.append({
            "diagnosis": k, "n": int(counts.get(k, 0)),
            "cs_mean_abs": float(sub["cs_mean"].mean()) if len(sub) else float("nan"),
            "cs_min_abs": float(sub["cs_mean"].min()) if len(sub) else float("nan"),
            "cs_max_abs": float(sub["cs_mean"].max()) if len(sub) else float("nan"),
            "sensitivity_mean": float(sub["sensitivity"].mean()) if len(sub) else float("nan"),
        })
    pd.DataFrame(quad_rows).to_csv(results_dir / "diagnosis_quadrant_summary.csv", index=False)

    print("\n=== Diagnosis distribution by pool ===")
    pivot = df.groupby(["pool", "diagnosis"]).size().unstack(fill_value=0)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
