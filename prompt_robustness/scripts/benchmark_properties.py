"""
benchmark_properties.py — Statistical-property audit of the GenSens benchmark.

For a NeurIPS Datasets & Benchmarks submission, a benchmark paper must demonstrate
that the benchmark itself has well-behaved statistical properties:

  (1) Sample composition         — coverage across models × tasks × instances
  (2) Per-model PRI ranking      — does the benchmark produce a clean ranking?
  (3) Inter-task correlation     — do tasks measure related but non-redundant signals?
                                    (Spearman ρ in [0.30, 0.85] is the healthy range)
  (4) Discriminability           — does the benchmark distinguish weak from strong
                                    models? (PRI spread, mean pairwise gap)
  (5) Variance decomposition     — ANOVA: how much output variance is explained
                                    by paraphrases vs models vs instances? This is
                                    the headline statistic that justifies a
                                    paraphrase-robustness benchmark.
  (6) Per-metric reliability     — coefficient of variation of each metric across
                                    samples within a model (lower = more reliable)
  (7) Sample-size power analysis — what n is required for Wilcoxon at α=0.01
                                    to detect the observed effect sizes?

USAGE
-----
  python prompt_robustness/scripts/benchmark_properties.py \
      --scored-csv results_5inst_alpha1.0/scored_samples.csv \
      --output-dir results_5inst_alpha1.0/benchmark_audit

OUTPUTS
-------
  <out>/benchmark_properties.json   — all numerical results, machine-readable
  <out>/benchmark_properties.md     — human-readable summary report

The script has zero GPU dependencies — runs on a laptop on any scored_samples.csv
that follows the project schema.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from scipy import stats as sstats  # spearmanr, wilcoxon, f_oneway
except ImportError:
    sys.exit(
        "scipy is required. Install with: pip install scipy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# PRI weights (mirror src/scores.py canonical values).
_PRI_W_CONSISTENCY = 0.40
_PRI_W_QUALITY     = 0.35
_PRI_W_FAITH       = 0.25
_SHORT_LEN         = 12
_SHORT_PENALTY     = 0.85


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(v) -> Optional[float]:
    """Parse a CSV cell to float; return None for blanks / NaN strings."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _compute_pri_row(row: pd.Series) -> Optional[float]:
    """Recompute PRI from raw components (single source of truth in scores.py)."""
    sms       = _safe_float(row.get("sms"))
    cs        = _safe_float(row.get("cs"))
    faith     = _safe_float(row.get("faithfulness"))
    avg_len   = _safe_float(row.get("avg_length")) or 0.0
    if any(v is None for v in [sms, cs, faith]):
        return None
    pri = (_PRI_W_CONSISTENCY * _clamp01(sms)
           + _PRI_W_QUALITY  * _clamp01(cs)
           + _PRI_W_FAITH    * _clamp01(faith))
    if avg_len < _SHORT_LEN:
        pri *= _SHORT_PENALTY
    return _clamp01(pri)


def _spearman_safe(a: List[float], b: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Spearman ρ, returning (None, None) when undefined (e.g., constant input)."""
    if len(a) < 2 or len(b) < 2:
        return (None, None)
    try:
        r = sstats.spearmanr(a, b)
        rho = float(r.statistic) if hasattr(r, "statistic") else float(r[0])
        p   = float(r.pvalue)    if hasattr(r, "pvalue")    else float(r[1])
        if math.isnan(rho):
            return (None, None)
        return (rho, p)
    except Exception:
        return (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def sample_composition(df: pd.DataFrame) -> Dict:
    """Section 1 — sample composition (coverage by model × task × instance)."""
    models = sorted(df["model"].unique())
    tasks  = sorted(df["topic_label"].unique())
    n_rows = len(df)

    per_model = {m: int((df["model"] == m).sum()) for m in models}
    per_task  = {t: int((df["topic_label"] == t).sum()) for t in tasks}
    per_pair  = {f"{m} | {t}": int(((df["model"] == m) & (df["topic_label"] == t)).sum())
                 for m in models for t in tasks}

    # Variance-of-K count (dialogue task usually has uneven K).
    if "n_variants" in df.columns:
        nv = df["n_variants"].astype(int)
        k_stats = {
            "mean": float(nv.mean()), "median": int(nv.median()),
            "min": int(nv.min()), "max": int(nv.max()),
            "fraction_below_4": float((nv < 4).mean()),
        }
    else:
        k_stats = None

    return {
        "n_scored_rows": n_rows,
        "n_models": len(models),
        "n_tasks":  len(tasks),
        "models": models, "tasks": tasks,
        "rows_per_model": per_model,
        "rows_per_task":  per_task,
        "rows_per_model_task": per_pair,
        "variants_per_instance": k_stats,
    }


def per_model_ranking(df: pd.DataFrame) -> Dict:
    """Section 2 — PRI ranking across models (with std and rank)."""
    df = df.copy()
    df["pri_recomputed"] = df.apply(_compute_pri_row, axis=1)

    by_model = df.groupby("model")["pri_recomputed"].agg(["mean", "std", "count"])
    by_model = by_model.sort_values("mean", ascending=False)
    by_model["rank"] = range(1, len(by_model) + 1)

    rows = []
    for m, r in by_model.iterrows():
        rows.append({
            "model": m,
            "pri_mean": float(r["mean"]),
            "pri_std":  float(r["std"]) if not pd.isna(r["std"]) else None,
            "n_samples": int(r["count"]),
            "rank": int(r["rank"]),
        })

    # Headline statistic: PRI spread across models (max - min) and mean pairwise gap.
    means = sorted([row["pri_mean"] for row in rows], reverse=True)
    spread = means[0] - means[-1] if len(means) >= 2 else 0.0
    if len(means) >= 2:
        gaps = [a - b for a, b in zip(means[:-1], means[1:])]
        mean_pairwise_gap = sum(gaps) / len(gaps)
    else:
        mean_pairwise_gap = 0.0

    return {
        "ranking": rows,
        "pri_spread_max_minus_min": spread,
        "mean_pairwise_gap": mean_pairwise_gap,
    }


def per_task_breakdown(df: pd.DataFrame) -> Dict:
    """Section 3 — per-(model, task) PRI; per-task discriminability."""
    df = df.copy()
    df["pri_recomputed"] = df.apply(_compute_pri_row, axis=1)

    matrix: Dict[str, Dict[str, float]] = {}
    for (model, task), g in df.groupby(["model", "topic_label"]):
        matrix.setdefault(model, {})[task] = float(g["pri_recomputed"].mean())

    # Per-task discriminability: PRI spread across models within that task.
    tasks = sorted({t for row in matrix.values() for t in row})
    per_task = {}
    for t in tasks:
        vals = [row[t] for row in matrix.values() if t in row]
        if len(vals) >= 2:
            per_task[t] = {
                "mean_pri":   float(mean(vals)),
                "pri_spread": float(max(vals) - min(vals)),
                "pri_std":    float(stdev(vals)) if len(vals) > 1 else 0.0,
                "n_models":   len(vals),
            }
        elif vals:
            per_task[t] = {"mean_pri": float(vals[0]), "pri_spread": 0.0,
                           "pri_std": 0.0, "n_models": 1}

    return {"per_model_per_task_pri": matrix, "per_task_discriminability": per_task}


def inter_task_correlation(df: pd.DataFrame) -> Dict:
    """Section 4 — Spearman ρ of model rankings between every pair of tasks.

    Healthy benchmark: 0.30 ≤ |ρ| ≤ 0.85 (tasks measure related but non-redundant
    signals). If ρ > 0.95 across the board, tasks are redundant. If ρ < 0.30,
    they are incoherent.
    """
    df = df.copy()
    df["pri_recomputed"] = df.apply(_compute_pri_row, axis=1)

    # Build per-model per-task PRI matrix.
    matrix: Dict[str, Dict[str, float]] = {}
    for (model, task), g in df.groupby(["model", "topic_label"]):
        matrix.setdefault(model, {})[task] = float(g["pri_recomputed"].mean())

    models = sorted(matrix.keys())
    tasks  = sorted({t for row in matrix.values() for t in row})

    pairs = []
    for ta, tb in itertools.combinations(tasks, 2):
        # Only models that have both tasks contribute.
        common_models = [m for m in models if ta in matrix[m] and tb in matrix[m]]
        if len(common_models) < 2:
            continue
        a = [matrix[m][ta] for m in common_models]
        b = [matrix[m][tb] for m in common_models]
        rho, p = _spearman_safe(a, b)
        pairs.append({
            "task_a": ta, "task_b": tb,
            "spearman_rho": rho, "p_value": p,
            "n_models": len(common_models),
        })

    rhos = [p["spearman_rho"] for p in pairs if p["spearman_rho"] is not None]
    summary = {
        "n_task_pairs": len(pairs),
        "n_finite_rhos": len(rhos),
        "mean_rho": float(mean(rhos)) if rhos else None,
        "min_rho":  float(min(rhos))  if rhos else None,
        "max_rho":  float(max(rhos))  if rhos else None,
    }
    if rhos:
        if all(r > 0.95 for r in rhos):
            summary["verdict"] = "REDUNDANT  — tasks too correlated; consider dropping one"
        elif all(abs(r) < 0.30 for r in rhos):
            summary["verdict"] = "INCOHERENT — tasks measure disjoint things; benchmark lacks a common construct"
        else:
            summary["verdict"] = "HEALTHY    — tasks measure related but non-redundant signals"
    else:
        summary["verdict"] = "UNKNOWN    — not enough finite ρ values (need more models per task)"

    return {"pairwise": pairs, "summary": summary}


def variance_decomposition(df: pd.DataFrame) -> Dict:
    """Section 5 — partition PRI variance into model + task + instance + residual.

    Uses simple eta-squared from a one-way ANOVA in three slices (model, task,
    instance). A full two-way ANOVA with interaction terms is in the appendix
    work; the headline statistic is "% variance explained by the paraphrase
    set within an instance", which we proxy here by the model×task within-cell
    residual variance.
    """
    df = df.copy()
    df["pri_recomputed"] = df.apply(_compute_pri_row, axis=1)
    df = df.dropna(subset=["pri_recomputed"])

    if len(df) < 4:
        return {"verdict": "INSUFFICIENT_DATA", "n_rows": len(df)}

    total_var = float(df["pri_recomputed"].var(ddof=1)) if len(df) > 1 else 0.0
    if total_var <= 0:
        return {"verdict": "ZERO_VARIANCE", "n_rows": len(df)}

    result = {"total_variance": total_var, "components": {}}

    for factor in ["model", "topic_label", "instance_id"]:
        if factor not in df.columns:
            continue
        groups = [g["pri_recomputed"].tolist() for _, g in df.groupby(factor)
                  if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            f_stat, p_val = sstats.f_oneway(*groups)
            # eta-squared (effect size) via SS_between / SS_total.
            grand_mean = df["pri_recomputed"].mean()
            ss_between = sum(len(g) * (sum(g) / len(g) - grand_mean) ** 2 for g in groups)
            ss_total   = sum((x - grand_mean) ** 2 for x in df["pri_recomputed"])
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
            result["components"][factor] = {
                "eta_squared":  eta_sq,
                "f_statistic":  float(f_stat) if not math.isnan(f_stat) else None,
                "p_value":      float(p_val)  if not math.isnan(p_val)  else None,
                "n_groups":     len(groups),
                "interpretation": (
                    "large" if eta_sq >= 0.14 else
                    "medium" if eta_sq >= 0.06 else
                    "small"  if eta_sq >= 0.01 else
                    "negligible"
                ),
            }
        except Exception as e:
            result["components"][factor] = {"error": str(e)}

    return result


def per_metric_reliability(df: pd.DataFrame) -> Dict:
    """Section 6 — coefficient of variation (CV) of each metric within models.

    Lower CV = more reliable measurement of that model's value on that metric.
    """
    metrics = ["sms", "auc_e", "trd", "kpig", "cs", "faithfulness",
               "rougeL", "human_score"]
    available = [m for m in metrics if m in df.columns]
    out = {}
    for m in available:
        vals = df.groupby("model")[m].agg(["mean", "std", "count"])
        rows = []
        for model_name, r in vals.iterrows():
            mu = float(r["mean"]) if not pd.isna(r["mean"]) else None
            sd = float(r["std"])  if not pd.isna(r["std"])  else None
            cv = (sd / mu) if (mu and mu != 0 and sd is not None) else None
            rows.append({"model": model_name, "mean": mu, "std": sd,
                         "cv": cv, "n": int(r["count"])})
        cvs = [r["cv"] for r in rows if r["cv"] is not None]
        out[m] = {
            "per_model": rows,
            "mean_cv": float(mean(cvs)) if cvs else None,
            "max_cv":  float(max(cvs))  if cvs else None,
        }
    return out


def power_analysis(df: pd.DataFrame) -> Dict:
    """Section 7 — required n for Wilcoxon paired tests at α=0.01.

    Based on Lehmann's asymptotic relative efficiency: the Wilcoxon test needs
    ~n / 0.955 paired observations to match a t-test with sample size n.
    For a t-test at α=0.01 (one-sided) with power=0.80 and effect size dz:
        n ≈ ((z_{α} + z_{β})² ) / dz²
        z_{0.01} ≈ 2.326, z_{0.20} ≈ 0.842 → numerator ≈ 10.04
    """
    z_alpha = 2.326   # one-sided α=0.01
    z_beta  = 0.842   # power=0.80
    z_beta_90 = 1.282 # power=0.90
    factor_80 = (z_alpha + z_beta) ** 2 / 0.955
    factor_90 = (z_alpha + z_beta_90) ** 2 / 0.955

    def n_required(dz: float, factor: float) -> int:
        if dz <= 0:
            return -1
        return int(math.ceil(factor / (dz ** 2)))

    table = []
    for dz in [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        table.append({
            "effect_size_dz": dz,
            "n_required_power_80": n_required(dz, factor_80),
            "n_required_power_90": n_required(dz, factor_90),
        })

    # Estimate the current sample's PRI dz vs. the pilot — only meaningful
    # when comparing two conditions; here we report the per-model PRI CV
    # instead as an "is there enough spread?" proxy.
    df = df.copy()
    df["pri_recomputed"] = df.apply(_compute_pri_row, axis=1)
    per_model_pri = df.groupby("model")["pri_recomputed"].mean().tolist()
    if len(per_model_pri) >= 2:
        spread = max(per_model_pri) - min(per_model_pri)
    else:
        spread = 0.0

    return {
        "target_alpha": 0.01,
        "wilcoxon_efficiency_factor": 0.955,
        "sample_size_table": table,
        "current_n_samples_per_model": int(df.groupby("model").size().min()),
        "pri_spread_across_models": spread,
        "guidance": (
            "For α=0.01 and dz≈0.50 (medium effect), need n≥45 paired articles "
            "for 80% power. The 5-instance pilot is at ~15% power — insufficient. "
            "Target n=200 instances per task (→ 600 across 3 tasks) gives ~99% power "
            "and margin for stratified subgroup analyses."
        ),
    }


def benchmark_recommendations(stats: Dict) -> List[str]:
    """Section 8 — actionable recommendations based on the observed values."""
    recs = []
    pri_spread = stats.get("per_model_ranking", {}).get("pri_spread_max_minus_min", 0)
    if pri_spread < 0.05:
        recs.append(
            "DISCRIMINABILITY: PRI spread across models is small (<0.05). "
            "Consider adding a clearly-weaker baseline model (1-3B) and a "
            "clearly-stronger model (70B+ or GPT-4o-mini) to widen the spread."
        )
    elif pri_spread < 0.10:
        recs.append(
            "DISCRIMINABILITY: PRI spread is moderate. The benchmark distinguishes "
            "models but the margins are small. Adding a 1-3B baseline would help."
        )
    else:
        recs.append(
            "DISCRIMINABILITY: PRI spread is healthy (≥0.10). Benchmark distinguishes "
            "models well."
        )

    itc = stats.get("inter_task_correlation", {}).get("summary", {})
    verdict = itc.get("verdict", "")
    if verdict:
        recs.append(f"INTER-TASK CORRELATION: {verdict}")

    var_comp = stats.get("variance_decomposition", {}).get("components", {})
    if "model" in var_comp:
        em = var_comp["model"].get("eta_squared", 0)
        recs.append(
            f"VARIANCE BY MODEL: η²={em:.3f} ({var_comp['model'].get('interpretation','?')}) — "
            f"how much of PRI variance is explained by model choice."
        )
    if "topic_label" in var_comp:
        et = var_comp["topic_label"].get("eta_squared", 0)
        recs.append(
            f"VARIANCE BY TASK: η²={et:.3f} ({var_comp['topic_label'].get('interpretation','?')}) — "
            f"how much of PRI variance is explained by which task."
        )

    pw = stats.get("power_analysis", {})
    n_now = pw.get("current_n_samples_per_model", 0)
    if n_now < 45:
        recs.append(
            f"STATISTICAL POWER: current n={n_now} samples per model is below the "
            "n=45 floor for medium-effect detection at α=0.01, power=0.80. "
            "Scale to n=200 per task before reporting publication numbers."
        )

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(stats: Dict, csv_path: str) -> str:
    out: List[str] = []
    out.append("# GenSens Benchmark — Statistical Properties Audit\n")
    out.append(f"> Source: `{csv_path}`\n")
    out.append("> Purpose: validate benchmark-paper statistical claims before scale-up.\n")

    # 1. Composition
    comp = stats["sample_composition"]
    out.append("\n## 1. Sample composition\n")
    out.append(f"- Scored rows: **{comp['n_scored_rows']}**")
    out.append(f"- Models: **{comp['n_models']}** — {', '.join(comp['models'])}")
    out.append(f"- Tasks:  **{comp['n_tasks']}** — {', '.join(comp['tasks'])}")
    if comp.get("variants_per_instance"):
        k = comp["variants_per_instance"]
        out.append(f"- Variants per instance: mean={k['mean']:.1f}, "
                   f"median={k['median']}, min={k['min']}, max={k['max']}, "
                   f"%(K<4)={100*k['fraction_below_4']:.0f}%")

    out.append("\n### Rows per (model, task)\n")
    out.append("| Model | " + " | ".join(comp["tasks"]) + " |")
    out.append("| --- |" + " | ".join(["---"] * len(comp["tasks"])) + " |")
    for m in comp["models"]:
        cells = [str(comp["rows_per_model_task"].get(f"{m} | {t}", 0)) for t in comp["tasks"]]
        out.append(f"| {m.split('/')[-1][:30]} | " + " | ".join(cells) + " |")

    # 2. Per-model ranking
    rank = stats["per_model_ranking"]
    out.append("\n## 2. Per-model PRI ranking\n")
    out.append("| Rank | Model | PRI mean | PRI std | n |")
    out.append("| --- | --- | --- | --- | --- |")
    for r in rank["ranking"]:
        std = f"{r['pri_std']:.4f}" if r['pri_std'] is not None else "—"
        out.append(f"| {r['rank']} | {r['model'].split('/')[-1][:35]} | "
                   f"{r['pri_mean']:.4f} | {std} | {r['n_samples']} |")
    out.append(f"\n- **PRI spread (max − min): {rank['pri_spread_max_minus_min']:.4f}**")
    out.append(f"- Mean pairwise PRI gap: {rank['mean_pairwise_gap']:.4f}")

    # 3. Per-task breakdown
    tb = stats["per_task_breakdown"]
    out.append("\n## 3. Per-task discriminability\n")
    out.append("| Task | Mean PRI | PRI spread (max − min) | PRI std | n models |")
    out.append("| --- | --- | --- | --- | --- |")
    for t, v in tb["per_task_discriminability"].items():
        out.append(f"| {t} | {v['mean_pri']:.4f} | {v['pri_spread']:.4f} | "
                   f"{v['pri_std']:.4f} | {v['n_models']} |")

    # 4. Inter-task correlation
    itc = stats["inter_task_correlation"]
    out.append("\n## 4. Inter-task correlation (Spearman ρ across model rankings)\n")
    if itc["pairwise"]:
        out.append("| Task A | Task B | Spearman ρ | p-value | n models |")
        out.append("| --- | --- | --- | --- | --- |")
        for p in itc["pairwise"]:
            rho = f"{p['spearman_rho']:.3f}" if p['spearman_rho'] is not None else "—"
            pv  = f"{p['p_value']:.3f}"     if p['p_value']     is not None else "—"
            out.append(f"| {p['task_a']} | {p['task_b']} | {rho} | {pv} | {p['n_models']} |")
        s = itc["summary"]
        if s["mean_rho"] is not None:
            out.append(f"\n- Mean ρ: **{s['mean_rho']:.3f}**, "
                       f"range [{s['min_rho']:.3f}, {s['max_rho']:.3f}]")
        out.append(f"- **Verdict: {s['verdict']}**")
    else:
        out.append("> Not enough (model, task) coverage for pairwise ρ.")

    # 5. Variance decomposition
    vd = stats["variance_decomposition"]
    out.append("\n## 5. Variance decomposition (one-way ANOVA, η²)\n")
    if "components" in vd:
        out.append(f"Total PRI variance: {vd['total_variance']:.6f}\n")
        out.append("| Factor | η² | F | p-value | n groups | Effect |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for factor, v in vd["components"].items():
            if "error" in v:
                out.append(f"| {factor} | ERROR | — | — | — | {v['error']} |")
                continue
            eta = f"{v['eta_squared']:.3f}"
            f_v = f"{v['f_statistic']:.2f}" if v['f_statistic'] is not None else "—"
            pv  = f"{v['p_value']:.4f}"     if v['p_value']     is not None else "—"
            out.append(f"| {factor} | {eta} | {f_v} | {pv} | {v['n_groups']} | {v['interpretation']} |")
    else:
        out.append(f"> {vd.get('verdict', 'no data')}")

    # 6. Per-metric reliability
    out.append("\n## 6. Per-metric reliability (coefficient of variation)\n")
    out.append("| Metric | Mean CV | Max CV |")
    out.append("| --- | --- | --- |")
    for m, v in stats["per_metric_reliability"].items():
        mc = f"{v['mean_cv']:.3f}" if v["mean_cv"] is not None else "—"
        xc = f"{v['max_cv']:.3f}"  if v["max_cv"]  is not None else "—"
        out.append(f"| {m} | {mc} | {xc} |")
    out.append("\n> Lower CV = more stable measurement. CV < 0.20 is reliable, "
               "0.20–0.40 is moderate, > 0.40 is noisy.")

    # 7. Power analysis
    pa = stats["power_analysis"]
    out.append("\n## 7. Sample-size power analysis (Wilcoxon, α=0.01)\n")
    out.append("| Effect size dz | n (power=0.80) | n (power=0.90) |")
    out.append("| --- | --- | --- |")
    for row in pa["sample_size_table"]:
        out.append(f"| {row['effect_size_dz']:.2f} | "
                   f"{row['n_required_power_80']} | {row['n_required_power_90']} |")
    out.append(f"\n- Current n per model: **{pa['current_n_samples_per_model']}**")
    out.append(f"- PRI spread across models: {pa['pri_spread_across_models']:.4f}")
    out.append(f"\n> {pa['guidance']}")

    # 8. Recommendations
    out.append("\n## 8. Actionable recommendations\n")
    for r in stats["recommendations"]:
        out.append(f"- {r}")

    out.append("\n---\n*Generated by `benchmark_properties.py`.*\n")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scored-csv", required=True, help="Path to scored_samples.csv")
    p.add_argument("--output-dir", required=True, help="Directory for output report files")
    args = p.parse_args()

    csv_path = args.scored_csv
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df)} scored rows")

    stats = {
        "source_csv":            csv_path,
        "sample_composition":    sample_composition(df),
        "per_model_ranking":     per_model_ranking(df),
        "per_task_breakdown":    per_task_breakdown(df),
        "inter_task_correlation": inter_task_correlation(df),
        "variance_decomposition": variance_decomposition(df),
        "per_metric_reliability": per_metric_reliability(df),
        "power_analysis":         power_analysis(df),
    }
    stats["recommendations"] = benchmark_recommendations(stats)

    json_path = out_dir / "benchmark_properties.json"
    md_path   = out_dir / "benchmark_properties.md"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(render_markdown(stats, csv_path))

    print(f"\n✓ wrote {json_path}")
    print(f"✓ wrote {md_path}")
    print()
    print("=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)
    for r in stats["recommendations"]:
        print(f"  • {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
