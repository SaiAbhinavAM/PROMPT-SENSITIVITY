#!/usr/bin/env python3
"""
aggregate_scores.py — composite scores + hierarchical aggregation for the
crossed-design GenSens summarization benchmark.

LEVEL 1 (per cell):
  Sensitivity = 0.30*sms_drift_n + 0.20*cs_var_n + 0.20*faith_var_n
              + 0.15*ppl_var_n + 0.15*pc_stab_var_n
  (spread metrics min-max normalized across ALL cells FIRST; if PPL/PC were
  not computed, the remaining weights are renormalized to sum to 1 — see
  method.md 2026-07-06 entry.)

  PRI = 0.40*sms_similarity + 0.35*cs_mean + 0.25*faith_mean
        (x0.85 if mean_output_len < 12 words)
  PRI is quality-gated ROBUSTNESS, not pure sensitivity — Sensitivity above
  is the headline sensitivity measure for this benchmark.

LEVEL 2 (per seed): mean/std/95% bootstrap CI across the N_ARTICLES shared
articles, computed cell-first (never flatten across articles before this
point).

LEVEL 3 (per pool, per dimension): Pool A and Pool B are kept separate in
every aggregate; never merged into one headline number.

Pure numpy/pandas/scipy — no GPU deps, so this (and diagnosis_matrix.py /
significance.py) can be run and smoke-tested without a GPU.
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
log = logging.getLogger("aggregate_scores")

SPREAD_METRICS = ["sms_drift", "cs_var", "faith_var", "ppl_var", "pc_stab_var"]
BASE_WEIGHTS = {"sms_drift": 0.30, "cs_var": 0.20, "faith_var": 0.20, "ppl_var": 0.15, "pc_stab_var": 0.15}

PRI_W_SMS = 0.40
PRI_W_CS = 0.35
PRI_W_FAITH = 0.25
SHORT_OUTPUT_LEN = 12
SHORT_OUTPUT_PENALTY = 0.85


def compute_sensitivity_and_pri(cells: list, normalization: str = "rank") -> list:
    """Mutates a copy of each cell dict in place, adding 'sensitivity' and
    'pri'. Normalizes each spread metric across ALL cells BEFORE combining.

    normalization:
      "rank"   — outlier-robust percentile normalization (DEFAULT). Prevents a
                 single heavy-tailed cell from collapsing 70–94% of cells to ≈0
                 (which made the min-max composite's nominal weights meaningless
                 for cs_var/faith_var/pc_stab_var — method.md 2026-07-06 fix).
      "minmax" — legacy empirical min-max (kept for ablation/reference).
    """
    cells = [dict(c) for c in cells]
    norm_fn = common.rank_normalize if normalization == "rank" else common.min_max_normalize

    normalized = {}
    active_metrics = []
    for metric in SPREAD_METRICS:
        raw = [c[metric] for c in cells]
        n_missing = sum(1 for v in raw if v is None)
        if n_missing == len(raw):
            log.warning(f"{metric}: not computed for any cell — excluded from Sensitivity, weight renormalized.")
            continue
        if n_missing > 0:
            log.warning(
                f"{metric}: missing for {n_missing}/{len(raw)} cells — excluding this metric entirely from "
                f"Sensitivity (partial availability would make the composite inconsistent across cells)."
            )
            continue
        normalized[metric] = norm_fn(raw)
        active_metrics.append(metric)

    weight_sum = sum(BASE_WEIGHTS[m] for m in active_metrics)
    if weight_sum <= 0:
        raise ValueError("No spread metrics available to compute Sensitivity.")
    renorm_weights = {m: BASE_WEIGHTS[m] / weight_sum for m in active_metrics}
    log.info(f"Sensitivity normalization: {normalization}; active weights: "
             + ", ".join(f"{m}={renorm_weights[m]:.3f}" for m in active_metrics))

    for i, c in enumerate(cells):
        sensitivity = sum(renorm_weights[m] * normalized[m][i] for m in active_metrics)
        c["sensitivity"] = float(sensitivity)

        pri = PRI_W_SMS * c["sms_similarity"] + PRI_W_CS * c["cs_mean"] + PRI_W_FAITH * c["faith_mean"]
        if c["mean_output_len"] < SHORT_OUTPUT_LEN:
            pri *= SHORT_OUTPUT_PENALTY
        c["pri"] = float(max(0.0, min(1.0, pri)))

    _log_composite_transparency(cells, normalized, active_metrics)
    return cells, active_metrics


def _log_composite_transparency(cells, normalized, active_metrics):
    """Make the composite honest and auditable: report how much each metric
    actually drives the Sensitivity ranking (Spearman of the normalized
    component vs the composite), and how close the full composite is to using
    the single strongest metric (sms_drift) alone. If the composite ≈ sms_drift
    (high Spearman), say so plainly rather than implying five independent
    signals."""
    from scipy.stats import spearmanr
    sens = np.array([c["sensitivity"] for c in cells], float)
    log.info("Composite transparency — Spearman(normalized component, Sensitivity):")
    for m in active_metrics:
        comp = np.array(normalized[m], float)
        rho = spearmanr(comp, sens).statistic
        log.info(f"    {m:<12s} rho={rho:.3f}")
    if "sms_drift" in active_metrics:
        rho_sms = spearmanr(np.array(normalized["sms_drift"], float), sens).statistic
        log.info(f"  Sensitivity ranking vs sms_drift-alone: Spearman rho={rho_sms:.3f} "
                 f"({'sms_drift dominates — composite is a mild refinement' if rho_sms > 0.9 else 'composite adds signal beyond sms_drift'})")


def level2_per_seed(cells: list) -> pd.DataFrame:
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c["seed_id"], []).append(c)

    rows = []
    for seed_id, group in sorted(by_seed.items()):
        pool = group[0]["pool"]
        dimension = group[0]["dimension"]
        sens_vals = [g["sensitivity"] for g in group]
        pri_vals = [g["pri"] for g in group]
        sens_ci = common.bootstrap_ci(sens_vals)
        pri_ci = common.bootstrap_ci(pri_vals)
        rows.append({
            "seed_id": seed_id, "pool": pool, "dimension": dimension,
            "n_articles": len(group),
            "sensitivity_mean": float(np.mean(sens_vals)),
            "sensitivity_std": float(np.std(sens_vals)),
            "sensitivity_ci_low": sens_ci[0], "sensitivity_ci_high": sens_ci[1],
            "pri_mean": float(np.mean(pri_vals)),
            "pri_std": float(np.std(pri_vals)),
            "pri_ci_low": pri_ci[0], "pri_ci_high": pri_ci[1],
            "sms_drift_mean": float(np.mean([g["sms_drift"] for g in group])),
            "cs_var_mean": float(np.mean([g["cs_var"] for g in group])),
            "faith_var_mean": float(np.mean([g["faith_var"] for g in group])),
            "cs_mean": float(np.mean([g["cs_mean"] for g in group])),
            "faith_mean": float(np.mean([g["faith_mean"] for g in group])),
        })
    return pd.DataFrame(rows).sort_values("seed_id").reset_index(drop=True)


def level3_by_group(cells: list, group_key: str) -> pd.DataFrame:
    by_group = {}
    for c in cells:
        by_group.setdefault(c[group_key], []).append(c)

    rows = []
    for key, group in sorted(by_group.items()):
        sens_vals = [g["sensitivity"] for g in group]
        pri_vals = [g["pri"] for g in group]
        sens_ci = common.bootstrap_ci(sens_vals)
        pri_ci = common.bootstrap_ci(pri_vals)
        rows.append({
            group_key: key,
            "n_cells": len(group),
            "sensitivity_mean": float(np.mean(sens_vals)),
            "sensitivity_std": float(np.std(sens_vals)),
            "sensitivity_ci_low": sens_ci[0], "sensitivity_ci_high": sens_ci[1],
            "pri_mean": float(np.mean(pri_vals)),
            "pri_std": float(np.std(pri_vals)),
            "pri_ci_low": pri_ci[0], "pri_ci_high": pri_ci[1],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--normalization", choices=["rank", "minmax"], default="rank",
                     help="Spread-metric normalization for the Sensitivity composite. 'rank' "
                          "(default, outlier-robust) is recommended; 'minmax' is the legacy "
                          "behavior kept for ablation.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT
    cells = common.read_jsonl(results_dir / "cell_metrics.jsonl")
    if not cells:
        raise FileNotFoundError(f"No cell metrics found in {results_dir}. Run compute_cell_metrics.py first.")
    log.info(f"Loaded {len(cells)} cells.")

    cells, active_metrics = compute_sensitivity_and_pri(cells, normalization=args.normalization)
    log.info(f"Sensitivity composite uses: {active_metrics}")

    # Persist scored cells (adds sensitivity/pri) alongside the raw cell metrics.
    common.write_jsonl(results_dir / "cell_metrics_scored.jsonl", cells)

    by_seed = level2_per_seed(cells)
    by_seed.to_csv(results_dir / "scores_by_seed.csv", index=False)
    log.info(f"Wrote {len(by_seed)} rows to scores_by_seed.csv")

    by_pool = level3_by_group(cells, "pool")
    by_pool.to_csv(results_dir / "scores_by_pool.csv", index=False)
    log.info(f"Wrote {len(by_pool)} rows to scores_by_pool.csv")

    # Pool-A gets its 10 dimensions and Pool-B its 4 categories from the same
    # "dimension" field — the spec's "14 dim/category rows" fall out of one
    # groupby since dimension values don't collide across pools.
    by_dimension = level3_by_group(cells, "dimension")
    pool_of_dim = {c["dimension"]: c["pool"] for c in cells}
    by_dimension.insert(1, "pool", by_dimension["dimension"].map(pool_of_dim))
    by_dimension.to_csv(results_dir / "scores_by_dimension.csv", index=False)
    log.info(f"Wrote {len(by_dimension)} rows to scores_by_dimension.csv")


if __name__ == "__main__":
    main()
