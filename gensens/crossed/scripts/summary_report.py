#!/usr/bin/env python3
"""
summary_report.py — final printed + saved report for the crossed-design
GenSens summarization prompt-sensitivity benchmark run.

Reads the outputs of run_inference_crossed.py, compute_cell_metrics.py,
aggregate_scores.py, diagnosis_matrix.py and significance.py; produces
results/summary_report.md and prints the same content to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def fmt_ci(mean, low, high):
    return f"{mean:.4f}  [95% CI {low:.4f}, {high:.4f}]"


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT

    responses = common.read_jsonl(results_dir / "responses_crossed.jsonl")
    cells = common.read_jsonl(results_dir / "cell_metrics_scored.jsonl")
    by_seed = pd.read_csv(results_dir / "scores_by_seed.csv")
    by_pool = pd.read_csv(results_dir / "scores_by_pool.csv")
    sig_path = results_dir / "significance.json"
    significance = json.loads(sig_path.read_text()) if sig_path.exists() else {}
    diag_path = results_dir / "diagnosis.csv"
    diagnosis = pd.read_csv(diag_path) if diag_path.exists() else None

    n_articles = len(set(r["article_id"] for r in responses)) if responses else 0
    n_seeds = len(set(r["seed_id"] for r in responses)) if responses else 0
    try:
        variants_per_seed_total = sum(len(s["variants"]) for s in common.load_seed_prompts())
    except Exception:
        variants_per_seed_total = common.N_SEEDS_EXPECTED * common.N_VARIANTS_PER_SEED
    expected_total = n_articles * variants_per_seed_total

    lines = []
    lines.append("# GenSens Crossed-Design Summarization — Summary Report\n")

    lines.append("## 1. Run size")
    lines.append(f"- Total outputs generated: **{len(responses)}** "
                 f"(articles={n_articles}, seeds={n_seeds}, target variants/cell={common.N_VARIANTS_PER_SEED}; "
                 f"expected {expected_total})")
    lines.append(f"- Complete cells scored: **{len(cells)}** (expected {n_articles * common.N_SEEDS_EXPECTED})\n")

    lines.append("## 2. SENSITIVITY by pool (headline result)")
    lines.append("*Sensitivity = rank-normalized composite (relative WITHIN this run — a value of "
                 "0.5 means 'median cell here', not an absolute level). `SBERT drift (abs)` is the "
                 "raw 1−mean-pairwise-cosine anchor: run-independent and comparable across models/runs. "
                 "`Resid` = Sensitivity with quality (cs/faith) regressed out (#8) — a mean-zero "
                 "'pure spread' with the quality confound removed.*")
    has_anchor = "sms_drift_mean" in by_pool.columns
    has_resid = "sensitivity_resid_mean" in by_pool.columns
    hdr = "| Pool | n_cells | Sensitivity mean | 95% CI |"
    if has_anchor:
        hdr += " SBERT drift (abs) |"
    if has_resid:
        hdr += " Resid (quality-adj) |"
    lines.append(hdr)
    lines.append("|---|---|---|---|" + ("---|" if has_anchor else "") + ("---|" if has_resid else ""))
    for _, row in by_pool.iterrows():
        cells_str = (f"| {row['pool']} | {int(row['n_cells'])} | {row['sensitivity_mean']:.4f} | "
                     f"[{row['sensitivity_ci_low']:.4f}, {row['sensitivity_ci_high']:.4f}] |")
        if has_anchor:
            cells_str += f" {row['sms_drift_mean']:.4f} |"
        if has_resid:
            cells_str += f" {row['sensitivity_resid_mean']:+.4f} |"
        lines.append(cells_str)
    lines.append("")
    lines.append("> ⚠️ **Pool-B caveat (#9):** Pool B prompts add a *constraint* (e.g. 'no proper nouns', "
                 "'isolate one theme'). Some cross-paraphrase output variation there reflects the "
                 "*legitimate degrees of freedom in satisfying the constraint*, not model fragility — "
                 "constrained tasks simply admit more valid answers. Read Pool-B sensitivity as an "
                 "upper bound on fragility, not pure fragility.")
    lines.append("")

    lines.append("## 3. PRI (quality-gated robustness) by pool")
    lines.append("*Note: PRI is quality-gated robustness (good AND stable), not pure sensitivity. "
                 "Section 2's Sensitivity composite is the headline sensitivity measure.*\n")
    lines.append("| Pool | n_cells | PRI mean | 95% CI |")
    lines.append("|---|---|---|---|")
    for _, row in by_pool.iterrows():
        lines.append(f"| {row['pool']} | {int(row['n_cells'])} | {row['pri_mean']:.4f} | "
                     f"[{row['pri_ci_low']:.4f}, {row['pri_ci_high']:.4f}] |")
    lines.append("")

    top5_most = by_seed.sort_values("sensitivity_mean", ascending=False).head(5)
    top5_least = by_seed.sort_values("sensitivity_mean", ascending=True).head(5)
    lines.append("## 4. Top 5 MOST sensitive seeds")
    lines.append("| seed_id | pool | dimension | Sensitivity mean | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for _, row in top5_most.iterrows():
        lines.append(f"| {row['seed_id']} | {row['pool']} | {row['dimension']} | "
                     f"{row['sensitivity_mean']:.4f} | [{row['sensitivity_ci_low']:.4f}, {row['sensitivity_ci_high']:.4f}] |")
    lines.append("\n## 5. Top 5 LEAST sensitive seeds")
    lines.append("| seed_id | pool | dimension | Sensitivity mean | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for _, row in top5_least.iterrows():
        lines.append(f"| {row['seed_id']} | {row['pool']} | {row['dimension']} | "
                     f"{row['sensitivity_mean']:.4f} | [{row['sensitivity_ci_low']:.4f}, {row['sensitivity_ci_high']:.4f}] |")
    lines.append("")

    lines.append("## 6. DIMENSION EFFECT — the primary finding")
    de = significance.get("dimension_effect", {})
    dan = _load_json(results_dir / "dimension_analysis.json")
    mm = (dan or {}).get("mixed_model", {})
    if de:
        eta_d = de.get("eta2_dimension_cells"); eta_p = de.get("eta2_pool_cells")
        lines.append(f"- Instruction **dimension** explains **{eta_d:.1%}** of sensitivity variance vs "
                     f"**{eta_p:.1%}** for the Pool A/B split — i.e. {de.get('interpretation','')}.")
    if mm.get("converged"):
        sig = "significant" if mm["lrt_dimension_p"] < 0.05 else "not significant"
        lines.append(f"- **Mixed-effects test (headline, valid unit):** `sensitivity ~ dimension + (1|seed) + (1|article)` "
                     f"→ LRT χ²={mm['lrt_dimension_chi2']:.1f}, df={mm['lrt_dimension_df']}, "
                     f"**p={mm['lrt_dimension_p']:.4g}** ({sig}). Variance: seed={mm['var_seed']:.4f}, "
                     f"article={mm['var_article']:.4f}, residual={mm['var_residual']:.4f}.")
    if de.get("kruskal_seedlevel_p") is not None:
        lines.append(f"  - _seed-aggregated Kruskal-Wallis (cruder, underpowered — 50 seeds / 14 dims):_ "
                     f"H={de['kruskal_seedlevel_H']:.1f}, p={de['kruskal_seedlevel_p']:.4g}")
    if dan and dan.get("residualized"):
        rz = dan["residualized"]
        lines.append(f"  - _robust to quality:_ dimension ranking is Spearman "
                     f"{rz['spearman_dim_ranking_raw_vs_resid']:.2f} between raw and "
                     f"quality-residualized Sensitivity (not a quality artifact).")
    if dan and dan.get("dimension_robustness"):
        dr = dan["dimension_robustness"]
        thin = ", ".join(dr.get("thin_dimensions_dropped", [])) or "none"
        p_kept = dr.get("mixed_lrt_p_kept")
        p_str = f"{p_kept:.4g}" if p_kept is not None else "n/a"
        lines.append(f"  - _robust to thin dimensions:_ dropping single-seed dimensions ({thin}) leaves "
                     f"η²={dr['eta2_dimension_kept']:.3f} (vs {dr['eta2_dimension_all']:.3f} with all), "
                     f"mixed-model p={p_str} — {dr.get('interpretation','')}.")
    lines.append("")
    lines.append("### Per-dimension sensitivity ranking")
    dim_rows = significance.get("dimension_ranking", [])
    lines.append("| Rank | Dimension | n_seeds | n_cells | Sensitivity mean | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(dim_rows, 1):
        lines.append(f"| {i} | {r['dimension']} | {r.get('n_seeds','?')} | {r.get('n_cells', r.get('n','?'))} | "
                     f"{r['sensitivity_mean']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines.append("")

    lines.append("## 7. Diagnosis matrix (quality x sensitivity)")
    if diagnosis is not None:
        counts = diagnosis["diagnosis"].value_counts()
        for k in ["True Robustness", "Fragile", "Consistently Poor", "Unreliable"]:
            lines.append(f"- {k}: {counts.get(k, 0)}")
    else:
        lines.append("- diagnosis.csv not found.")
    lines.append("")

    lines.append("## 8. Pool A vs B (secondary — pool explains little variance)")
    pc = significance.get("pool_comparison")  # seed-level = valid unit
    if pc:
        lines.append(f"- Seed-level Mann-Whitney U: n_A={pc['n_pool_a']} seeds, n_B={pc['n_pool_b']} seeds, "
                     f"U={pc['u_statistic']:.1f}, p={pc['p_value']:.4g}, "
                     f"rank-biserial r={pc['rank_biserial_effect_size']:.4f} ({pc['direction']}). "
                     f"Pool is NOT the primary axis — see §6; it explains ~10x less variance than dimension.")
    else:
        lines.append("- Pool A vs Pool B (seed-level): not run (insufficient seeds per pool).")
    pcr = significance.get("pool_comparison_percell_ref")
    if pcr:
        lines.append(f"  - _per-cell (pseudoreplicated, reference only — anticonservative, not for inference):_ "
                     f"p={pcr['p_value']:.4g}")
    mvl = significance.get("most_vs_least_sensitive_seed")
    if mvl:
        lines.append(f"- **Most ({mvl['most_sensitive_seed']}) vs least ({mvl['least_sensitive_seed']}) sensitive "
                     f"seed (Wilcoxon):** W={mvl['wilcoxon_statistic']:.1f}, p={mvl['p_value']:.4g}, "
                     f"dz={mvl['cohens_dz']:.4f}, n_paired={mvl['n_paired_articles']}")
    else:
        lines.append("- Most-vs-least sensitive seed: not run (insufficient paired articles).")
    lines.append("")

    lines.append("## 9. Warnings")
    all_warnings = list(significance.get("warnings", []))
    cc_warn_path = results_dir / "compute_cell_metrics_warnings.txt"
    if cc_warn_path.exists():
        all_warnings.extend([w for w in cc_warn_path.read_text().splitlines() if w.strip()])
    if all_warnings:
        for w in all_warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None.")
    lines.append("")

    report = "\n".join(lines)
    out_path = results_dir / "summary_report.md"
    out_path.write_text(report)
    print(report)
    print(f"\n[summary_report.py] Wrote {out_path}")


if __name__ == "__main__":
    main()
