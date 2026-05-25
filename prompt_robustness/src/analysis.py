"""
analysis.py — Post-benchmark correlation and statistical analysis module.

This module provides functions for computing statistical correlations between
automated metrics and human scores, as well as aggregate analysis utilities.

Used after the main benchmark pipeline completes to generate research-grade
diagnostic outputs (saved to results/correlations.json).

Mathematical References:
- Pearson r: Measures linear correlation between two variables.
- Spearman ρ: Measures monotonic (rank-order) correlation.
"""

import json
import os
import math
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


def compute_correlation_pair(
    x: List[float],
    y: List[float],
    x_name: str = "X",
    y_name: str = "Y"
) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlation between two score vectors.

    Handles edge cases:
        - Constant vectors (zero variance) → correlation undefined → returns 0.0
        - Mismatched lengths → raises ValueError
        - Too few data points (< 3) → returns 0.0 for both

    Args:
        x: First score vector.
        y: Second score vector.
        x_name: Name label for x (used in logging).
        y_name: Name label for y (used in logging).

    Returns:
        Dict with 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p',
        'n_samples', and 'pair_name' keys.
    """
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {x_name} has {len(x)}, {y_name} has {len(y)}")

    n = len(x)
    result = {
        'pair_name': f"{x_name}_vs_{y_name}",
        'n_samples': n,
        'pearson_r': 0.0,
        'pearson_p': 1.0,
        'spearman_rho': 0.0,
        'spearman_p': 1.0,
    }

    if n < 3:
        logger.warning(f"Too few samples ({n}) for reliable correlation: {x_name} vs {y_name}")
        return result

    # Check for constant vectors (zero variance breaks correlation)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        logger.warning(f"Near-constant vector detected for {x_name} vs {y_name}, returning 0.0")
        return result

    try:
        p_corr, p_pval = pearsonr(x, y)
        if math.isnan(p_corr):
            p_corr = 0.0
            p_pval = 1.0

        s_corr, s_pval = spearmanr(x, y)
        if math.isnan(s_corr):
            s_corr = 0.0
            s_pval = 1.0

        result.update({
            'pearson_r': round(float(p_corr), 4),
            'pearson_p': round(float(p_pval), 6),
            'spearman_rho': round(float(s_corr), 4),
            'spearman_p': round(float(s_pval), 6),
        })

    except Exception as e:
        logger.error(f"Correlation computation failed for {x_name} vs {y_name}: {e}")

    return result


def compute_full_correlation_analysis(all_results: List[Dict]) -> Dict:
    """
    Compute comprehensive correlation analysis across all benchmark results.

    Computes Pearson and Spearman correlations between:
        - PRI vs Human_Score
        - SMS vs Human_Score
        - CS vs Human_Score
        - Final_Score vs Human_Score
        - TRD vs Human_Score (inverted: lower TRD should correlate with higher quality)

    Args:
        all_results: List of per-sample result dictionaries from the benchmark pipeline.

    Returns:
        Dict containing all correlation results, ready for JSON serialization.
    """
    if not all_results or len(all_results) < 3:
        logger.warning("Insufficient results for correlation analysis (need ≥ 3 samples)")
        return {
            'status': 'insufficient_data',
            'n_samples': len(all_results) if all_results else 0,
            'correlations': {}
        }

    # Extract score vectors from results
    pri_scores = [r.get('pri', 0.0) for r in all_results]
    human_scores = [r.get('human_score', 0.0) for r in all_results]
    cs_scores = [r.get('cs', 0.0) for r in all_results]
    faith_scores = [r.get('faithfulness', 0.0) for r in all_results]

    # Extract SMS from nested metrics dict
    sms_scores = [r.get('metrics', {}).get('sms', 0.0) for r in all_results]
    trd_scores = [r.get('metrics', {}).get('trd', 0.0) for r in all_results]

    # Advanced metrics (may not be present in all results)
    trd_sem_scores = [r.get('trd_semantic', 0.0) for r in all_results]
    kpig_adv_scores = [r.get('kpig_advanced', 0.0) for r in all_results]

    correlations = {}

    # Core correlations against the human score.
    # R5: Final_Score is NOT correlated against Human_Score here — Final_Score
    # CONTAINS Human_Score (0.4 weight), so that correlation is circular and
    # spuriously inflated. PRI_vs_Human is the honest validity signal.
    metric_pairs = [
        (pri_scores, human_scores, 'PRI', 'Human_Score'),
        (sms_scores, human_scores, 'SMS', 'Human_Score'),
        (cs_scores, human_scores, 'CS', 'Human_Score'),
        (faith_scores, human_scores, 'Faithfulness', 'Human_Score'),
        (trd_scores, human_scores, 'TRD', 'Human_Score'),
    ]

    # Advanced metric correlations (only if they contain non-zero values)
    if any(s != 0.0 for s in trd_sem_scores):
        metric_pairs.append((trd_sem_scores, human_scores, 'TRD_Semantic', 'Human_Score'))
    if any(s != 0.0 for s in kpig_adv_scores):
        metric_pairs.append((kpig_adv_scores, human_scores, 'KPIG_Advanced', 'Human_Score'))

    for x, y, x_name, y_name in metric_pairs:
        key = f"{x_name}_vs_{y_name}"
        correlations[key] = compute_correlation_pair(x, y, x_name, y_name)

    return {
        'status': 'complete',
        'n_samples': len(all_results),
        'correlations': correlations,
    }


def save_correlation_results(
    correlations: Dict,
    output_dir: str = "results"
) -> str:
    """
    Save correlation analysis results to results/correlations.json.

    Args:
        correlations: The correlation analysis dictionary.
        output_dir: Directory to save the output file.

    Returns:
        str: Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "correlations.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=4, default=str)

    logger.info(f"Correlation results saved to {output_path}")
    return output_path


def compute_metric_summary_statistics(all_results: List[Dict]) -> Dict:
    """
    Compute aggregate summary statistics (mean, std, min, max, median) for all metrics.

    Useful for research reporting and identifying metric distribution characteristics.

    Args:
        all_results: List of per-sample result dictionaries.

    Returns:
        Dict mapping metric names to their summary statistics.
    """
    if not all_results:
        return {}

    # Collect all available numeric fields
    metric_keys = ['pri', 'cs', 'hs_score', 'consistency', 'human_score',
                   'final_score', 'avg_length', 'avg_coverage', 'confidence',
                   'sms_wasserstein', 'trd_semantic', 'kpig_advanced', 'usd']

    summary = {}
    for key in metric_keys:
        values = [r.get(key, None) for r in all_results]
        values = [v for v in values if v is not None and isinstance(v, (int, float))]

        if not values:
            continue

        arr = np.array(values)
        summary[key] = {
            'mean': round(float(np.mean(arr)), 4),
            'std': round(float(np.std(arr)), 4),
            'min': round(float(np.min(arr)), 4),
            'max': round(float(np.max(arr)), 4),
            'median': round(float(np.median(arr)), 4),
            'n': len(values),
        }

    # Also extract nested metrics (sms, trd, kpig, etc.)
    nested_keys = ['sms', 'auc_e', 'trd', 'kpig', 'ppl_var', 'bf']
    for key in nested_keys:
        values = [r.get('metrics', {}).get(key, None) for r in all_results]
        values = [v for v in values if v is not None and isinstance(v, (int, float))]

        if not values:
            continue

        arr = np.array(values)
        summary[f"metric_{key}"] = {
            'mean': round(float(np.mean(arr)), 4),
            'std': round(float(np.std(arr)), 4),
            'min': round(float(np.min(arr)), 4),
            'max': round(float(np.max(arr)), 4),
            'median': round(float(np.median(arr)), 4),
            'n': len(values),
        }

    return summary
