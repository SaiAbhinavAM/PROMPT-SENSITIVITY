#!/usr/bin/env python3
"""
significance.py — statistical tests on per-cell Sensitivity distributions
(n = N_ARTICLES per seed, since the crossed design shares the same article
set across all seeds).

A. Pool A vs Pool B: Mann-Whitney U on per-cell Sensitivity + rank-biserial
   effect size. Expected: Pool B (rephrasing-under-constraint) more
   sensitive than Pool A (pure rephrasing).
B. Per-dimension ranking: mean Sensitivity with 95% bootstrap CI per
   dimension; most/least sensitive identified.
C. Most- vs least-sensitive seed: Wilcoxon signed-rank on paired
   per-article Sensitivity (valid pairing because every seed sees every
   article).

Tests only run where n >= 20 per group; warns if N_ARTICLES < 30 (the floor
stated in the spec for reliable statistics).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("significance")

MIN_N_FOR_TEST = 20      # per-cell / per-article units (grow with N_ARTICLES)
MIN_SEEDS_FOR_TEST = 10  # per-seed unit: fixed by the dataset (35 vs 15 seeds),
                         # never grows with N_ARTICLES, so it needs a lower bar.
                         # Mann-Whitney is valid at 15 vs 35; power is the only
                         # concern and is flagged separately.
N_ARTICLES_FLOOR = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT
    cells = common.read_jsonl(results_dir / "cell_metrics_scored.jsonl")
    if not cells:
        raise FileNotFoundError(f"No scored cell metrics found in {results_dir}. Run aggregate_scores.py first.")

    n_articles = len(set(c["article_id"] for c in cells))
    warnings = []
    if n_articles < N_ARTICLES_FLOOR:
        msg = f"N_ARTICLES={n_articles} is below the floor of {N_ARTICLES_FLOOR} — significance results may be underpowered."
        log.warning(msg)
        warnings.append(msg)

    out = {"n_articles": n_articles, "warnings": warnings}

    # ── A. Pool A vs Pool B: Mann-Whitney U ─────────────────────────────
    # HEADLINE test is at the SEED level. Per-cell Sensitivity values are NOT
    # independent — Pool A is 35 seeds × N_ARTICLES cells and Pool B is 15 ×
    # N_ARTICLES, and cells sharing a seed (same prompt) or article (same
    # content) are correlated. A per-cell Mann-Whitney treats those correlated
    # cells as independent samples, which is pseudoreplication and yields an
    # anticonservative (too-small) p-value. The valid unit is the per-seed mean
    # sensitivity (seeds are the independently sampled prompts). We therefore
    # report the seed-level test as the primary result and keep the per-cell
    # test only as a clearly-labeled, non-inferential reference.
    #   (method.md 2026-07-06 pseudoreplication fix.)

    def _mwu(b_vals, a_vals, unit, min_n):
        if len(a_vals) >= min_n and len(b_vals) >= min_n:
            u_stat, p_val = stats.mannwhitneyu(b_vals, a_vals, alternative="two-sided")
            res = {
                "unit": unit,
                "n_pool_a": len(a_vals), "n_pool_b": len(b_vals),
                "u_statistic": float(u_stat), "p_value": float(p_val),
                "rank_biserial_effect_size": common.rank_biserial_from_u(u_stat, len(b_vals), len(a_vals)),
                "pool_b_median": float(np.median(b_vals)), "pool_a_median": float(np.median(a_vals)),
                "direction": "Pool B more sensitive" if np.median(b_vals) > np.median(a_vals) else "Pool A more sensitive",
            }
            if min(len(a_vals), len(b_vals)) < 20:
                res["low_power_warning"] = (f"smaller group has {min(len(a_vals), len(b_vals))} units; "
                                            f"test is valid but underpowered.")
            return res
        msg = (f"Pool comparison ({unit}) skipped: n_pool_a={len(a_vals)}, n_pool_b={len(b_vals)} "
               f"(need >= {min_n} each).")
        log.warning(msg)
        warnings.append(msg)
        return None

    # Seed-level: aggregate each seed to its mean sensitivity across the shared articles.
    seed_means = {}
    for c in cells:
        seed_means.setdefault((c["seed_id"], c["pool"]), []).append(c["sensitivity"])
    seed_a = [float(np.mean(v)) for (s, pl), v in seed_means.items() if pl == "A"]
    seed_b = [float(np.mean(v)) for (s, pl), v in seed_means.items() if pl == "B"]
    pool_seed = _mwu(seed_b, seed_a, "per_seed", MIN_SEEDS_FOR_TEST)

    # Per-cell: pseudoreplicated — reference only, must NOT be used for inference.
    cell_a = [c["sensitivity"] for c in cells if c["pool"] == "A"]
    cell_b = [c["sensitivity"] for c in cells if c["pool"] == "B"]
    pool_cell = _mwu(cell_b, cell_a, "per_cell", MIN_N_FOR_TEST)
    if pool_cell is not None:
        pool_cell["note"] = ("PSEUDOREPLICATED — cells sharing a seed/article are correlated; "
                             "p-value is anticonservative. Not for inference; use per_seed.")

    out["pool_comparison"] = pool_seed              # headline (valid unit)
    out["pool_comparison_percell_ref"] = pool_cell  # reference only
    if seed_a and seed_b:
        out["pool_comparison_note"] = (
            f"Headline uses per-seed means (n_A={len(seed_a)} seeds, n_B={len(seed_b)} seeds). "
            f"Per-cell test retained under pool_comparison_percell_ref for reference only.")

    # ── B0. DIMENSION EFFECT (the headline result) ───────────────────────
    # The Pool A/B split explains almost none of the sensitivity variance; the
    # instruction DIMENSION explains most of it. We quantify that two ways:
    #   (i)  eta^2 (variance explained) for dimension vs pool — descriptive.
    #   (ii) a SEED-AGGREGATED Kruskal-Wallis across dimensions — the valid,
    #        non-pseudoreplicated test (each seed contributes ONE value = its
    #        mean sensitivity across the shared articles), mirroring the
    #        seed-level pool test. Per-cell KW would be pseudoreplicated.
    sens_all = np.array([c["sensitivity"] for c in cells], float)
    grand = sens_all.mean()
    ss_total = float(((sens_all - grand) ** 2).sum())

    def _eta2(group_key):
        groups = {}
        for c in cells:
            groups.setdefault(c[group_key], []).append(c["sensitivity"])
        ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
        return float(ss_between / ss_total) if ss_total > 0 else float("nan")

    # seed-level means, tagged with each seed's (single) dimension and pool
    seed_dim = {}
    seed_pool = {}
    seed_sens = {}
    for c in cells:
        seed_sens.setdefault(c["seed_id"], []).append(c["sensitivity"])
        seed_dim[c["seed_id"]] = c["dimension"]
        seed_pool[c["seed_id"]] = c["pool"]
    seed_mean = {s: float(np.mean(v)) for s, v in seed_sens.items()}

    dim_groups_seed = {}
    for s, m in seed_mean.items():
        dim_groups_seed.setdefault(seed_dim[s], []).append(m)
    # KW needs >=2 groups each with >=1 seed; only test groups that have members
    kw_groups = [np.array(v) for v in dim_groups_seed.values() if len(v) >= 1]
    dim_effect = {
        "eta2_dimension_cells": _eta2("dimension"),
        "eta2_pool_cells": _eta2("pool"),
        "interpretation": "dimension explains ~%.0fx more sensitivity variance than pool"
                          % ((_eta2("dimension") / _eta2("pool")) if _eta2("pool") > 0 else float("nan")),
    }
    if len(kw_groups) >= 2 and all(len(g) >= 1 for g in kw_groups):
        try:
            H, p_kw = stats.kruskal(*kw_groups)
            dim_effect["kruskal_seedlevel_H"] = float(H)
            dim_effect["kruskal_seedlevel_p"] = float(p_kw)
            dim_effect["kruskal_note"] = ("seed-aggregated (each of the %d seeds = 1 value); "
                                          "valid non-pseudoreplicated dimension test" % len(seed_mean))
        except Exception as e:
            dim_effect["kruskal_error"] = str(e)[:120]
    out["dimension_effect"] = dim_effect

    # ── B. Per-dimension ranking ─────────────────────────────────────────
    by_dim = {}
    for c in cells:
        by_dim.setdefault(c["dimension"], []).append(c["sensitivity"])
    dim_rows = []
    for dim, vals in by_dim.items():
        ci = common.bootstrap_ci(vals)
        seed_n = sum(1 for s in seed_dim if seed_dim[s] == dim)
        dim_rows.append({
            "dimension": dim, "n_cells": len(vals), "n_seeds": seed_n,
            "sensitivity_mean": float(np.mean(vals)),
            "ci_low": ci[0], "ci_high": ci[1],
        })
    dim_rows.sort(key=lambda r: r["sensitivity_mean"], reverse=True)
    out["dimension_ranking"] = dim_rows
    out["most_sensitive_dimension"] = dim_rows[0]["dimension"] if dim_rows else None
    out["least_sensitive_dimension"] = dim_rows[-1]["dimension"] if dim_rows else None

    # ── C. Most- vs least-sensitive seed: Wilcoxon signed-rank ──────────
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c["seed_id"], {})[c["article_id"]] = c["sensitivity"]
    seed_means = {seed_id: float(np.mean(list(arts.values()))) for seed_id, arts in by_seed.items()}
    if len(seed_means) >= 2:
        most_seed = max(seed_means, key=seed_means.get)
        least_seed = min(seed_means, key=seed_means.get)
        common_articles = sorted(set(by_seed[most_seed]) & set(by_seed[least_seed]))
        if len(common_articles) >= MIN_N_FOR_TEST:
            most_vals = [by_seed[most_seed][a] for a in common_articles]
            least_vals = [by_seed[least_seed][a] for a in common_articles]
            diffs = np.array(most_vals) - np.array(least_vals)
            if np.all(diffs == 0):
                w_stat, p_val = float("nan"), 1.0
            else:
                w_stat, p_val = stats.wilcoxon(most_vals, least_vals)
            dz = common.cohens_dz(diffs.tolist())
            out["most_vs_least_sensitive_seed"] = {
                "most_sensitive_seed": most_seed, "most_sensitive_mean": seed_means[most_seed],
                "least_sensitive_seed": least_seed, "least_sensitive_mean": seed_means[least_seed],
                "n_paired_articles": len(common_articles),
                "wilcoxon_statistic": float(w_stat), "p_value": float(p_val),
                "cohens_dz": dz,
            }
        else:
            msg = f"Most-vs-least seed test skipped: only {len(common_articles)} paired articles (need >= {MIN_N_FOR_TEST})."
            log.warning(msg)
            warnings.append(msg)
            out["most_vs_least_sensitive_seed"] = None
    else:
        out["most_vs_least_sensitive_seed"] = None

    out["warnings"] = warnings
    out_path = results_dir / "significance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Wrote significance results to {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
