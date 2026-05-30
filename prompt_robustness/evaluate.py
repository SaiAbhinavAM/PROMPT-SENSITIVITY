#!/usr/bin/env python3
"""
Evaluation Script for LL-PIRC Pipeline (Phase 4).

Loads baseline and PIRC results, computes:
1. Relative variance reduction: 1 - mean(var_pirc) / mean(var_baseline)
2. Mean ROUGE-L change (baseline vs PIRC)
3. Wilcoxon signed-rank test on paired variance/ROUGE-L values
4. Summary table
5. S(ℓ) sensitivity curve plots for sample articles

Saves:
- results/eval_summary.json
- results/layer_sensitivity_plot.png

Reference: method_analysis_prompt_sensitivity.md, Section 5 (lines 753-756)

Usage:
    python evaluate.py [--config config.yaml]
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np

# ─── Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("evaluate")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# Load Results
# ═══════════════════════════════════════════════════════════════════════════════

def load_results(results_dir: str) -> tuple:
    """Load baseline and PIRC results."""
    results_path = Path(results_dir)

    baseline_path = results_path / "baseline.json"
    pirc_path = results_path / "pirc.json"

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline results not found at {baseline_path}. "
            f"Run experiment_baseline.py first."
        )
    if not pirc_path.exists():
        raise FileNotFoundError(
            f"PIRC results not found at {pirc_path}. "
            f"Run experiment_pirc.py first."
        )

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    with open(pirc_path, 'r') as f:
        pirc = json.load(f)

    logger.info(
        f"Loaded baseline ({len(baseline['results'])} articles) "
        f"and PIRC ({len(pirc['results'])} articles) results"
    )

    return baseline, pirc


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_wilcoxon_test(
    var_baseline: np.ndarray,
    var_pirc: np.ndarray,
    significance_level: float = 0.01
) -> Dict:
    """
    Run Wilcoxon signed-rank test on paired variance values.

    The Wilcoxon signed-rank test is non-parametric and appropriate for
    paired samples that may not be normally distributed.

    Args:
        var_baseline: Array of baseline variances per article.
        var_pirc: Array of PIRC variances per article.
        significance_level: p-value threshold for significance.

    Returns:
        result: Dict with test statistic, p-value, and significance.
    """
    from scipy.stats import wilcoxon

    # Remove pairs where both are zero (no difference)
    diff = var_baseline - var_pirc
    nonzero_mask = diff != 0

    if nonzero_mask.sum() < 2:
        logger.warning(
            "Too few non-zero differences for Wilcoxon test. "
            "Returning N/A."
        )
        return {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'note': 'Insufficient non-zero differences'
        }

    stat, p_value = wilcoxon(
        var_baseline[nonzero_mask],
        var_pirc[nonzero_mask],
        alternative='greater'  # one-sided: baseline > PIRC
    )

    result = {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < significance_level,
        'significance_level': significance_level,
        'n_pairs': int(nonzero_mask.sum()),
    }

    logger.info(
        f"Wilcoxon test: W={stat:.2f}, p={p_value:.6f}, "
        f"significant={result['significant']} "
        f"(α={significance_level}, n={result['n_pairs']})"
    )

    return result


def run_rouge_wilcoxon_test(
    rouge_baseline: np.ndarray,
    rouge_pirc: np.ndarray,
    significance_level: float = 0.01
) -> Dict:
    """
    Run Wilcoxon signed-rank test on ROUGE-L scores (two-sided).

    Tests whether PIRC significantly changes ROUGE-L quality.
    """
    from scipy.stats import wilcoxon

    diff = rouge_baseline - rouge_pirc
    nonzero_mask = diff != 0

    if nonzero_mask.sum() < 2:
        return {
            'statistic': None,
            'p_value': None,
            'significant': False,
            'note': 'Insufficient non-zero differences'
        }

    stat, p_value = wilcoxon(
        rouge_baseline[nonzero_mask],
        rouge_pirc[nonzero_mask],
        alternative='two-sided'
    )

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < significance_level,
        'significance_level': significance_level,
        'n_pairs': int(nonzero_mask.sum()),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap confidence interval for the mean of a paired difference."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {'mean': None, 'low': None, 'high': None, 'n': 0}

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        'mean': float(values.mean()),
        'low': float(np.quantile(means, tail)),
        'high': float(np.quantile(means, 1.0 - tail)),
        'confidence': confidence,
        'n': int(len(values)),
    }


def paired_effect_size_dz(before: np.ndarray, after: np.ndarray) -> Optional[float]:
    """Paired Cohen's dz for before-after differences."""
    diff = np.asarray(before, dtype=float) - np.asarray(after, dtype=float)
    if len(diff) < 2 or diff.std(ddof=1) < 1e-12:
        return None
    return float(diff.mean() / diff.std(ddof=1))


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_curves(
    pirc_results: List[Dict],
    num_samples: int = 5,
    output_path: str = "results/layer_sensitivity_plot.png",
    dpi: int = 150
):
    """
    Plot S(ℓ) sensitivity curves for sample articles.

    Each curve shows how the layer-wise sensitivity signal S(ℓ) varies
    across layers, with the identified sensitive layer ℓ* marked.

    Args:
        pirc_results: List of PIRC result dicts (must contain 'S_curve').
        num_samples: Number of sample articles to plot.
        output_path: Path to save the plot.
        dpi: DPI for the output image.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend

    # Select articles that have S_curve data
    valid_results = [
        r for r in pirc_results
        if r.get('S_curve') is not None and len(r.get('S_curve', {})) > 0
    ]

    if not valid_results:
        logger.warning("No valid S_curve data found. Skipping plot.")
        return

    num_to_plot = min(num_samples, len(valid_results))
    # Select evenly spaced samples
    indices = np.linspace(0, len(valid_results) - 1, num_to_plot, dtype=int)

    fig, axes = plt.subplots(1, num_to_plot, figsize=(5 * num_to_plot, 4),
                              squeeze=False)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_to_plot))

    for plot_idx, data_idx in enumerate(indices):
        ax = axes[0][plot_idx]
        result = valid_results[data_idx]

        S_curve = result['S_curve']
        ell_star = result.get('ell_star', None)

        # Sort by layer index
        layers = sorted([int(k) for k in S_curve.keys()])
        S_values = [S_curve[str(ell)] for ell in layers]

        ax.plot(layers, S_values, 'o-', color=colors[plot_idx],
                linewidth=2, markersize=4, alpha=0.8)

        # Mark ℓ*
        if ell_star is not None and str(ell_star) in S_curve:
            ax.axvline(x=ell_star, color='red', linestyle='--',
                      alpha=0.7, linewidth=1.5)
            ax.plot(ell_star, S_curve[str(ell_star)], 'r*',
                   markersize=15, zorder=5)

        ax.set_xlabel('Layer ℓ', fontsize=11)
        ax.set_ylabel('S(ℓ) = Var[PPL]', fontsize=11)
        ax.set_title(
            f"Article {result.get('article_idx', data_idx)}\n"
            f"ℓ*={ell_star}",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.suptitle(
        'Layer-wise Sensitivity Curves S(ℓ)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Sensitivity curves plot saved to {output_path}")


def plot_variance_comparison(
    var_baseline: np.ndarray,
    var_pirc: np.ndarray,
    output_path: str = "results/variance_comparison.png",
    dpi: int = 150
):
    """
    Plot baseline vs PIRC variance comparison.

    Creates a scatter plot and histogram of variance values.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: baseline vs PIRC variance
    ax1.scatter(var_baseline, var_pirc, alpha=0.6, s=40, c='steelblue',
                edgecolors='navy', linewidth=0.5)
    max_val = max(var_baseline.max(), var_pirc.max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1.5,
             label='y = x (no change)')
    ax1.set_xlabel('Baseline ROUGE-L Variance', fontsize=11)
    ax1.set_ylabel('PIRC ROUGE-L Variance', fontsize=11)
    ax1.set_title('Variance: Baseline vs PIRC', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Histogram of variance reduction
    reduction = var_baseline - var_pirc
    ax2.hist(reduction, bins=20, color='steelblue', edgecolor='navy',
             alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axvline(x=reduction.mean(), color='green', linestyle='-',
                alpha=0.7, linewidth=2, label=f'Mean = {reduction.mean():.4f}')
    ax2.set_xlabel('Variance Reduction (Baseline - PIRC)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Variance Reduction', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Variance comparison plot saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(eval_summary: Dict):
    """Print a formatted summary table of evaluation results."""
    print("\n" + "=" * 70)
    print("LL-PIRC EVALUATION SUMMARY")
    print("=" * 70)

    # Main metrics
    print(f"\n{'Metric':<40} {'Value':>15}")
    print("-" * 55)
    print(f"{'Articles evaluated':<40} {eval_summary['num_articles']:>15d}")
    print(f"{'Successful PIRC runs':<40} {eval_summary['num_successful']:>15d}")

    print(f"\n--- ROUGE-L ---")
    print(f"{'Baseline mean ROUGE-L':<40} {eval_summary['baseline_mean_rouge']:.4f}")
    print(f"{'PIRC mean ROUGE-L':<40} {eval_summary['pirc_mean_rouge']:.4f}")
    print(f"{'ROUGE-L change':<40} {eval_summary['rouge_change']:.4f}")
    print(f"{'ROUGE-L change (%)':<40} {eval_summary['rouge_change_pct']:.2f}%")

    print(f"\n--- Variance ---")
    print(f"{'Baseline mean variance':<40} {eval_summary['baseline_mean_var']:.6f}")
    print(f"{'PIRC mean variance':<40} {eval_summary['pirc_mean_var']:.6f}")
    print(
        f"{'Relative variance reduction':<40} "
        f"{eval_summary['relative_var_reduction']:.2%}"
    )
    var_ci = eval_summary.get('variance_reduction_ci', {})
    if var_ci.get('low') is not None:
        print(
            f"{'Variance reduction 95% CI':<40} "
            f"[{var_ci['low']:.6f}, {var_ci['high']:.6f}]"
        )
    if eval_summary.get('variance_effect_size_dz') is not None:
        print(
            f"{'Variance effect size dz':<40} "
            f"{eval_summary['variance_effect_size_dz']:.3f}"
        )

    print(f"\n--- Statistical Tests ---")
    var_test = eval_summary['wilcoxon_variance']
    if var_test['p_value'] is not None:
        print(
            f"{'Variance Wilcoxon W':<40} {var_test['statistic']:.2f}"
        )
        print(
            f"{'Variance Wilcoxon p-value':<40} {var_test['p_value']:.6f}"
        )
        sig_str = "YES ✓" if var_test['significant'] else "NO ✗"
        print(
            f"{'Significant (p < {:.3f})?'.format(var_test['significance_level']):<40} "
            f"{sig_str}"
        )
    else:
        print(f"{'Variance Wilcoxon test':<40} N/A")

    rouge_test = eval_summary.get('wilcoxon_rouge', {})
    if rouge_test.get('p_value') is not None:
        print(
            f"{'ROUGE-L Wilcoxon p-value':<40} {rouge_test['p_value']:.6f}"
        )
    rouge_ci = eval_summary.get('rouge_change_ci', {})
    if rouge_ci.get('low') is not None:
        print(
            f"{'ROUGE-L change 95% CI':<40} "
            f"[{rouge_ci['low']:.6f}, {rouge_ci['high']:.6f}]"
        )
    if eval_summary.get('rouge_effect_size_dz') is not None:
        print(
            f"{'ROUGE-L effect size dz':<40} "
            f"{eval_summary['rouge_effect_size_dz']:.3f}"
        )

    print(f"\n--- Sensitive Layer (ℓ*) ---")
    ell_stats = eval_summary.get('ell_star_stats', {})
    if ell_stats:
        print(f"{'ℓ* mean':<40} {ell_stats['mean']:.1f}")
        print(f"{'ℓ* std':<40} {ell_stats['std']:.1f}")
        print(f"{'ℓ* range':<40} [{ell_stats['min']}, {ell_stats['max']}]")
        print(
            f"{'ℓ* coefficient of variation':<40} "
            f"{ell_stats['cv']:.3f}"
        )

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(config: dict):
    """
    Run the full evaluation pipeline.

    1. Load baseline and PIRC results
    2. Compute variance reduction, ROUGE-L change
    3. Run Wilcoxon signed-rank tests
    4. Print summary table
    5. Generate plots
    6. Save eval_summary.json
    """
    results_dir = config['experiment']['results_dir']
    num_sample_plots = config['evaluation']['num_sample_plots']
    plot_dpi = config['evaluation']['plot_dpi']
    significance_level = config['evaluation']['significance_level']

    # ─── Load results ─────────────────────────────────────────────────────
    baseline, pirc = load_results(results_dir)

    # ─── Align results by article index ───────────────────────────────────
    baseline_by_idx = {r['article_idx']: r for r in baseline['results']}
    pirc_by_idx = {r['article_idx']: r for r in pirc['results']}

    # Find common articles with valid PIRC results
    common_indices = sorted(
        set(baseline_by_idx.keys()) & set(pirc_by_idx.keys())
    )
    valid_indices = [
        idx for idx in common_indices
        if pirc_by_idx[idx].get('pirc_rouge_var') is not None
        and (
            pirc_by_idx[idx].get('pirc_rouge_mean') is not None
            or pirc_by_idx[idx].get('pirc_rouge_l') is not None
        )
    ]

    logger.info(
        f"Common articles: {len(common_indices)}, "
        f"valid PIRC results: {len(valid_indices)}"
    )

    if not valid_indices:
        logger.error("No valid paired results. Cannot evaluate.")
        return

    # ─── Extract paired metrics ───────────────────────────────────────────
    var_baseline = np.array([
        baseline_by_idx[idx]['rouge_var'] for idx in valid_indices
    ])
    rouge_baseline_mean = np.array([
        baseline_by_idx[idx]['rouge_mean'] for idx in valid_indices
    ])
    rouge_pirc = np.array([
        pirc_by_idx[idx].get('pirc_rouge_mean', pirc_by_idx[idx].get('pirc_rouge_l'))
        for idx in valid_indices
    ])
    var_pirc = np.array([
        pirc_by_idx[idx]['pirc_rouge_var'] for idx in valid_indices
    ])

    # ─── Compute metrics ──────────────────────────────────────────────────
    baseline_mean_var = float(var_baseline.mean())
    pirc_mean_var = float(var_pirc.mean())

    if baseline_mean_var > 0:
        relative_var_reduction = 1.0 - pirc_mean_var / baseline_mean_var
    else:
        relative_var_reduction = 0.0

    baseline_mean_rouge = float(rouge_baseline_mean.mean())
    pirc_mean_rouge = float(rouge_pirc.mean())
    rouge_change = pirc_mean_rouge - baseline_mean_rouge
    rouge_change_pct = (
        (rouge_change / baseline_mean_rouge * 100)
        if baseline_mean_rouge > 0 else 0.0
    )

    # ─── Statistical tests ────────────────────────────────────────────────
    wilcoxon_var = run_wilcoxon_test(
        var_baseline, var_pirc, significance_level
    )
    wilcoxon_rouge = run_rouge_wilcoxon_test(
        rouge_baseline_mean, rouge_pirc, significance_level
    )
    variance_reduction_diff = var_baseline - var_pirc
    rouge_change_diff = rouge_pirc - rouge_baseline_mean
    variance_reduction_ci = bootstrap_mean_ci(variance_reduction_diff)
    rouge_change_ci = bootstrap_mean_ci(rouge_change_diff)
    variance_effect_size_dz = paired_effect_size_dz(var_baseline, var_pirc)
    # For ROUGE, positive means PIRC improved; paired_effect_size_dz uses
    # before-after, so invert the sign for interpretability.
    rouge_effect_size_dz_raw = paired_effect_size_dz(rouge_baseline_mean, rouge_pirc)
    rouge_effect_size_dz = (
        -rouge_effect_size_dz_raw
        if rouge_effect_size_dz_raw is not None else None
    )

    # ─── ℓ* statistics ────────────────────────────────────────────────────
    ell_stars = [
        pirc_by_idx[idx]['ell_star']
        for idx in valid_indices
        if 'ell_star' in pirc_by_idx[idx]
    ]

    ell_star_stats = {}
    if ell_stars:
        ell_arr = np.array(ell_stars)
        ell_star_stats = {
            'mean': float(ell_arr.mean()),
            'std': float(ell_arr.std()),
            'min': int(ell_arr.min()),
            'max': int(ell_arr.max()),
            'cv': float(ell_arr.std() / ell_arr.mean()) if ell_arr.mean() > 0 else 0.0,
            'values': [int(v) for v in ell_stars],
        }

    # ─── Build summary ────────────────────────────────────────────────────
    eval_summary = {
        'num_articles': len(valid_indices),
        'num_successful': len(valid_indices),
        'baseline_mean_rouge': baseline_mean_rouge,
        'pirc_mean_rouge': pirc_mean_rouge,
        'rouge_change': rouge_change,
        'rouge_change_pct': rouge_change_pct,
        'baseline_mean_var': baseline_mean_var,
        'pirc_mean_var': pirc_mean_var,
        'relative_var_reduction': relative_var_reduction,
        'wilcoxon_variance': wilcoxon_var,
        'wilcoxon_rouge': wilcoxon_rouge,
        'variance_reduction_ci': variance_reduction_ci,
        'rouge_change_ci': rouge_change_ci,
        'variance_effect_size_dz': variance_effect_size_dz,
        'rouge_effect_size_dz': rouge_effect_size_dz,
        'ell_star_stats': ell_star_stats,
        'interpretation': {
            'note': (
                "PIRC variance is measured across K PIRC-stabilized outputs "
                "for the same K paraphrase variants used by the baseline. "
                "It is no longer set to zero by construction."
            ),
            'variance_eliminated': pirc_mean_var == 0.0,
            'quality_preserved': abs(rouge_change) < 2.0,
        }
    }

    # ─── Print summary ────────────────────────────────────────────────────
    print_summary_table(eval_summary)

    # ─── Save summary ─────────────────────────────────────────────────────
    summary_path = Path(results_dir) / "eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(eval_summary, f, indent=2, default=str)
    logger.info(f"Evaluation summary saved to {summary_path}")

    # ─── Generate plots ──────────────────────────────────────────────────
    # S(ℓ) curves
    pirc_results_with_curves = [
        pirc_by_idx[idx] for idx in valid_indices
        if pirc_by_idx[idx].get('S_curve')
    ]

    if pirc_results_with_curves:
        plot_sensitivity_curves(
            pirc_results_with_curves,
            num_samples=num_sample_plots,
            output_path=str(Path(results_dir) / "layer_sensitivity_plot.png"),
            dpi=plot_dpi
        )

    # Variance comparison
    if len(var_baseline) > 0:
        plot_variance_comparison(
            var_baseline,
            var_pirc,
            output_path=str(Path(results_dir) / "variance_comparison.png"),
            dpi=plot_dpi
        )

    return eval_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LL-PIRC pipeline results"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Flaw §4.2 — deterministic seeding for reproducibility (bootstrap CI
    # resamples + any random tie-breaks).
    try:
        from src.utils import set_global_seed
        set_global_seed(int(config.get("seed", 42)) if isinstance(config, dict) else 42)
    except Exception:
        pass
    run_evaluation(config)


if __name__ == "__main__":
    main()
