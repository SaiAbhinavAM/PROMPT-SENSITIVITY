#!/usr/bin/env python3
"""
dimension_analysis.py — rigorous analysis of the dimension effect on prompt
sensitivity, plus a quality-residualized "pure sensitivity" measure.

Motivated by the 30-article result (method.md 2026-07-14): the instruction
DIMENSION explains ~10x more sensitivity variance than the Pool A/B split
(eta^2 0.18 vs 0.019), and Sensitivity is mildly entangled with quality
(r(Sensitivity, CS_mean) = -0.235). This script:

  1. RESIDUALIZED SENSITIVITY — regress Sensitivity on the quality covariates
     (CS_mean, faith_mean) via OLS and keep the residual as a quality-adjusted
     "pure spread" measure. Reports how much the dimension/seed ranking changes
     (Spearman raw-vs-residual): if it barely moves, the dimension finding is
     not a quality artifact.

  2. MIXED-EFFECTS MODEL — Sensitivity ~ C(dimension) with CROSSED random
     intercepts for seed and article: the proper test that the dimension effect
     is real beyond individual-seed idiosyncrasy and article effects. Reports
     variance components and a likelihood-ratio test vs an intercept-only model.

Pure CPU (statsmodels); no GPU. Reads cell_metrics_scored.jsonl.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dimension_analysis")


def residualize(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'sensitivity_resid' = Sensitivity with quality (CS_mean, faith_mean)
    regressed out via OLS. The residual is orthogonal to the quality covariates
    by construction, so it is a spread signal not confounded by 'unstable
    prompts are also slightly worse'."""
    import statsmodels.formula.api as smf
    ols = smf.ols("sensitivity ~ cs_mean + faith_mean", data=df).fit()
    df = df.copy()
    df["sensitivity_resid"] = df["sensitivity"] - ols.predict(df)
    return df, ols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT

    cells = common.read_jsonl(results_dir / "cell_metrics_scored.jsonl")
    if not cells:
        raise FileNotFoundError(f"No scored cells in {results_dir}. Run aggregate_scores.py first.")
    df = pd.DataFrame(cells)
    log.info(f"Loaded {len(df)} cells, {df['seed_id'].nunique()} seeds, {df['article_id'].nunique()} articles.")

    from scipy.stats import spearmanr
    out = {"n_cells": int(len(df)), "n_seeds": int(df["seed_id"].nunique()),
           "n_articles": int(df["article_id"].nunique())}

    # ── 1. Residualized (quality-adjusted) sensitivity ───────────────────
    df, ols = residualize(df)
    raw_dim = df.groupby("dimension")["sensitivity"].mean()
    res_dim = df.groupby("dimension")["sensitivity_resid"].mean()
    raw_seed = df.groupby("seed_id")["sensitivity"].mean()
    res_seed = df.groupby("seed_id")["sensitivity_resid"].mean()
    out["residualized"] = {
        "quality_r2": float(ols.rsquared),
        "note": "sensitivity_resid = Sensitivity with CS_mean + faith_mean regressed out",
        "spearman_dim_ranking_raw_vs_resid": float(spearmanr(raw_dim, res_dim).statistic),
        "spearman_seed_ranking_raw_vs_resid": float(spearmanr(raw_seed, res_seed).statistic),
        "dimension_ranking_residualized": [
            {"dimension": d, "sensitivity_resid_mean": float(v)}
            for d, v in res_dim.sort_values(ascending=False).items()
        ],
    }
    log.info("Residualized: dimension ranking raw-vs-resid Spearman = %.3f "
             "(high => dimension effect is not a quality artifact)"
             % out["residualized"]["spearman_dim_ranking_raw_vs_resid"])

    # ── 2. Mixed-effects model: Sensitivity ~ dimension + (1|seed)+(1|article)
    out["mixed_model"] = _fit_mixed(df)

    # ── 2b. Thin-dimension robustness: does the dimension effect survive dropping
    # dimensions backed by only ONE seed? A 1-seed dimension conflates "dimension"
    # with that single prompt, so the honest check is whether the effect holds
    # without them (method.md 2026-08-09 audit).
    out["dimension_robustness"] = _thin_dimension_robustness(df)

    out_path = results_dir / "dimension_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "residualized"}, indent=2))
    print("\nResidualized dimension ranking (quality-adjusted, top/bottom 3):")
    rr = out["residualized"]["dimension_ranking_residualized"]
    for r in rr[:3] + [{"dimension": "...", "sensitivity_resid_mean": 0.0}] + rr[-3:]:
        print("  %-22s %+.3f" % (r["dimension"], r["sensitivity_resid_mean"]))


def _eta2_dimension(df: pd.DataFrame) -> float:
    """Fraction of Sensitivity variance explained by dimension (between-group SS
    / total SS) on the given cells."""
    s = df["sensitivity"].to_numpy(dtype=float)
    grand = s.mean()
    ss_total = float(((s - grand) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    ss_between = sum(len(g) * (g["sensitivity"].mean() - grand) ** 2 for _, g in df.groupby("dimension"))
    return float(ss_between / ss_total)


def _thin_dimension_robustness(df: pd.DataFrame, min_seeds: int = 2) -> dict:
    """Re-run the dimension effect after dropping dimensions supported by fewer
    than `min_seeds` seeds (which conflate the dimension with one specific
    prompt). Reports eta^2 and the mixed-model LRT with and without them, plus
    the Spearman of the surviving dimensions' ranking between the two fits — if
    the effect and ranking hold, the dimension finding is not driven by a thin,
    single-prompt dimension."""
    from scipy.stats import spearmanr

    seeds_per_dim = df.groupby("dimension")["seed_id"].nunique()
    thin = sorted(seeds_per_dim[seeds_per_dim < min_seeds].index.tolist())
    kept = df[~df["dimension"].isin(thin)].copy()

    eta_all = _eta2_dimension(df)
    eta_kept = _eta2_dimension(kept) if kept["dimension"].nunique() >= 2 else float("nan")

    # Ranking stability on the dimensions retained in BOTH fits.
    common_dims = sorted(set(kept["dimension"]))
    rank_all = df[df["dimension"].isin(common_dims)].groupby("dimension")["sensitivity"].mean()
    rank_kept = kept.groupby("dimension")["sensitivity"].mean()
    rho = float(spearmanr(rank_all.loc[common_dims], rank_kept.loc[common_dims]).statistic) if len(common_dims) >= 2 else float("nan")

    mixed_kept = _fit_mixed(kept) if kept["dimension"].nunique() >= 2 else {"converged": False, "error": "too few dimensions after drop"}
    return {
        "min_seeds_threshold": min_seeds,
        "thin_dimensions_dropped": thin,
        "n_dimensions_all": int(df["dimension"].nunique()),
        "n_dimensions_kept": int(kept["dimension"].nunique()),
        "eta2_dimension_all": eta_all,
        "eta2_dimension_kept": eta_kept,
        "mixed_lrt_p_kept": mixed_kept.get("lrt_dimension_p"),
        "spearman_ranking_all_vs_kept": rho,
        "interpretation": ("dimension effect is ROBUST to dropping single-seed dimensions"
                           if (not np.isnan(eta_kept) and abs(eta_all - eta_kept) < 0.02) else
                           "dimension effect changes when thin dimensions are dropped — inspect"),
    }


def _fit_mixed(df: pd.DataFrame) -> dict:
    """Sensitivity ~ C(dimension) with crossed random intercepts for seed and
    article (statsmodels variance-components MixedLM). LRT vs intercept-only for
    an omnibus dimension test. Wrapped in try/except: if it fails to converge,
    we fall back to the seed-aggregated Kruskal-Wallis in significance.py."""
    try:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2
        d = df.copy()
        d["grp"] = 1  # single top-level group; seed & article enter as variance comps
        vc = {"seed": "0 + C(seed_id)", "article": "0 + C(article_id)"}

        # ML (not REML) so the fixed-effect LRT is valid.
        full = smf.mixedlm("sensitivity ~ C(dimension)", d, groups="grp", vc_formula=vc).fit(reml=False)
        reduced = smf.mixedlm("sensitivity ~ 1", d, groups="grp", vc_formula=vc).fit(reml=False)
        lr = 2.0 * (full.llf - reduced.llf)
        ddf = int(d["dimension"].nunique() - 1)
        p_lrt = float(chi2.sf(lr, ddf))

        # statsmodels stores vcomp as an array in the order of the vc_formula keys.
        names = list(vc.keys())
        vcomp = np.asarray(full.vcomp).ravel()
        vc_named = {names[i]: float(vcomp[i]) for i in range(min(len(names), len(vcomp)))}
        return {
            "converged": True,
            "lrt_dimension_chi2": float(lr),
            "lrt_dimension_df": ddf,
            "lrt_dimension_p": p_lrt,
            "var_seed": vc_named.get("seed"),
            "var_article": vc_named.get("article"),
            "var_residual": float(full.scale),
            "interpretation": ("dimension fixed effect is significant controlling for crossed "
                               "seed+article random intercepts" if p_lrt < 0.05 else
                               "dimension effect NOT significant once seed/article variance is modeled"),
            "note": ("Random-intercept variances: seed captures per-prompt idiosyncrasy WITHIN a "
                     "dimension; article captures content effects. A significant LRT means dimension "
                     "explains sensitivity beyond those."),
        }
    except Exception as e:
        log.warning(f"Mixed model did not fit ({type(e).__name__}: {str(e)[:100]}); "
                    f"rely on seed-aggregated Kruskal-Wallis in significance.json.")
        return {"converged": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}


if __name__ == "__main__":
    main()
