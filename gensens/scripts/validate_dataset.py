#!/usr/bin/env python3
"""
GenSens Dataset Validator
=========================
Runs post-generation quality checks on the GenSens dataset:
1. Completeness check
2. Similarity distribution
3. Diversity check (pairwise Levenshtein distance)
4. Length check
5. Exact copy check
6. Full prompt substitution check
7. Summary table

Usage:
    python scripts/validate_dataset.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Utility: Levenshtein word-level distance
# ─────────────────────────────────────────────────────────────

def word_levenshtein(s1: str, s2: str) -> int:
    """Compute word-level Levenshtein distance between two strings."""
    w1 = s1.split()
    w2 = s2.split()
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if w1[i - 1] == w2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


# ─────────────────────────────────────────────────────────────
# Validation checks
# ─────────────────────────────────────────────────────────────

def check_completeness(
    records: List[Dict[str, Any]],
    task_name: str,
    expected_instances: int = 200,
    expected_variants: int = 8,
) -> Tuple[List[str], List[str]]:
    """Check that every task has expected instances and variants."""
    warnings = []
    errors = []

    if len(records) != expected_instances:
        errors.append(
            f"[{task_name}] Expected {expected_instances} instances, got {len(records)}"
        )

    for rec in records:
        n_var = len(rec.get("variants", []))
        if n_var != expected_variants:
            warnings.append(
                f"[{task_name}] Instance {rec['instance_id']} has {n_var} variants "
                f"(expected {expected_variants})"
            )

    return warnings, errors


def check_similarity_distribution(
    records: List[Dict[str, Any]], task_name: str, threshold: float = 0.82
) -> Tuple[List[str], Dict[str, Any]]:
    """Check SBERT similarity distribution and flag below-threshold variants."""
    warnings = []
    all_sims = []

    for rec in records:
        for v in rec.get("variants", []):
            sim = v.get("sbert_similarity", 0.0)
            all_sims.append(sim)
            if sim < threshold:
                warnings.append(
                    f"[{task_name}] Instance {rec['instance_id']} variant {v['variant_idx']} — "
                    f"sim={sim:.4f} < {threshold}"
                )

    sims = np.array(all_sims) if all_sims else np.array([0.0])

    # Build histogram buckets
    bins = [0.0, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95, 1.01]
    hist, _ = np.histogram(sims, bins=bins)
    histogram = {
        f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(hist[i])
        for i in range(len(hist))
    }

    stats = {
        "mean": round(float(sims.mean()), 4),
        "std": round(float(sims.std()), 4),
        "min": round(float(sims.min()), 4),
        "max": round(float(sims.max()), 4),
        "histogram": histogram,
    }

    return warnings, stats


def check_diversity(
    records: List[Dict[str, Any]], task_name: str, min_mean_distance: int = 10
) -> List[str]:
    """Flag instances where variants are too similar (low mean pairwise edit distance)."""
    warnings = []

    for rec in records:
        variants = rec.get("variants", [])
        if len(variants) < 2:
            continue

        texts = [v["paraphrased_text"] for v in variants]
        distances = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                distances.append(word_levenshtein(texts[i], texts[j]))

        mean_dist = np.mean(distances) if distances else 0
        if mean_dist < min_mean_distance:
            warnings.append(
                f"[{task_name}] Instance {rec['instance_id']} — mean pairwise edit distance "
                f"= {mean_dist:.1f} < {min_mean_distance} (low diversity)"
            )

    return warnings


def check_length(
    records: List[Dict[str, Any]], task_name: str, min_words: int = 5
) -> List[str]:
    """Flag variants that are too short or too long."""
    warnings = []

    for rec in records:
        base_wc = len(rec["base_text"].split())
        max_wc = base_wc * 3

        for v in rec.get("variants", []):
            var_wc = len(v["paraphrased_text"].split())
            if var_wc < min_words:
                warnings.append(
                    f"[{task_name}] Instance {rec['instance_id']} variant {v['variant_idx']} — "
                    f"too short: {var_wc} words < {min_words}"
                )
            elif var_wc > max_wc:
                warnings.append(
                    f"[{task_name}] Instance {rec['instance_id']} variant {v['variant_idx']} — "
                    f"too long: {var_wc} words > 3× base ({max_wc})"
                )

    return warnings


def check_exact_copies(
    records: List[Dict[str, Any]], task_name: str
) -> List[str]:
    """Flag any variant that is an exact copy of the base_text."""
    warnings = []

    for rec in records:
        base = rec["base_text"].strip()
        for v in rec.get("variants", []):
            if v["paraphrased_text"].strip() == base:
                warnings.append(
                    f"[{task_name}] Instance {rec['instance_id']} variant {v['variant_idx']} — "
                    f"EXACT COPY of base_text"
                )

    return warnings


def check_full_prompt_substitution(
    records: List[Dict[str, Any]], task_name: str
) -> List[str]:
    """Verify that full_prompt contains paraphrased_text and NOT original base_text."""
    warnings = []

    for rec in records:
        base_text = rec["base_text"].strip()
        for v in rec.get("variants", []):
            full_prompt = v.get("full_prompt", "")
            para_text = v["paraphrased_text"].strip()

            if para_text not in full_prompt:
                warnings.append(
                    f"[{task_name}] Instance {rec['instance_id']} variant {v['variant_idx']} — "
                    f"full_prompt does NOT contain paraphrased_text"
                )

            # Check base_text is NOT in full_prompt (it should have been replaced)
            # But only if base_text != paraphrased_text (which we already check elsewhere)
            if base_text != para_text and base_text in full_prompt:
                warnings.append(
                    f"[{task_name}] Instance {rec['instance_id']} variant {v['variant_idx']} — "
                    f"full_prompt still contains original base_text (substitution failed)"
                )

    return warnings


def check_reference_present(
    records: List[Dict[str, Any]], task_name: str
) -> List[str]:
    """Flag any instance whose metadata lacks a non-empty reference_output.

    The new four-task set requires EVERY task to carry a gold reference, so an
    empty reference is a data-quality problem worth surfacing (warning).
    Reads the canonical ``reference_output`` key, with the legacy
    ``gold_summary`` alias as a fallback.
    """
    warnings = []

    for rec in records:
        meta = rec.get("metadata", {}) or {}
        reference = meta.get("reference_output", meta.get("gold_summary", ""))
        if not reference or not str(reference).strip():
            warnings.append(
                f"[{task_name}] Instance {rec['instance_id']} — "
                f"EMPTY reference_output (no gold output)"
            )

    return warnings


# ─────────────────────────────────────────────────────────────
# Main validation
# ─────────────────────────────────────────────────────────────

def validate_dataset(
    data_dir: str,
    expected_instances: int = 200,
    expected_variants: int = 8,
    tasks: List[str] = None,
):
    """Run all validation checks on generated dataset files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if tasks is None:
        tasks = ["summarization", "creative", "dialogue", "qa"]
    report = {}
    summary_table = []

    for task_name in tasks:
        # Find the output file (R7: pick the NEWEST matching file by mtime, so a
        # fresh run is validated instead of an arbitrary stale one).
        pattern = f"gensens_{task_name}_"
        matching_files = sorted(
            [
                f for f in os.listdir(data_dir)
                if f.startswith(pattern) and f.endswith(".jsonl") and "checkpoint" not in f
            ],
            key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
            reverse=True,
        )

        if not matching_files:
            logger.warning(f"No output file found for task '{task_name}' in {data_dir}")
            summary_table.append({
                "task": task_name,
                "instances": 0,
                "variants": 0,
                "passed": False,
                "warnings": 0,
                "errors": 1,
            })
            report[task_name] = {"status": "missing_file"}
            continue

        filepath = os.path.join(data_dir, matching_files[0])
        logger.info(f"\nValidating: {filepath}")

        # Load records
        records = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info(f"  Loaded {len(records)} records")

        task_warnings = []
        task_errors = []

        # 1. Completeness
        w, e = check_completeness(records, task_name, expected_instances, expected_variants)
        task_warnings.extend(w)
        task_errors.extend(e)
        logger.info(f"  [1] Completeness: {len(e)} errors, {len(w)} warnings")

        # 2. Similarity distribution
        w, sim_stats = check_similarity_distribution(records, task_name)
        task_warnings.extend(w)
        logger.info(f"  [2] Similarity: mean={sim_stats['mean']:.4f}, "
                     f"below-threshold warnings={len(w)}")
        logger.info(f"      Histogram: {sim_stats['histogram']}")

        # 3. Diversity check
        w = check_diversity(records, task_name)
        task_warnings.extend(w)
        logger.info(f"  [3] Diversity: {len(w)} low-diversity warnings")

        # 4. Length check
        w = check_length(records, task_name)
        task_warnings.extend(w)
        logger.info(f"  [4] Length: {len(w)} warnings")

        # 5. Exact copy check
        w = check_exact_copies(records, task_name)
        task_warnings.extend(w)
        logger.info(f"  [5] Exact copies: {len(w)} warnings")

        # 6. Full prompt substitution check
        w = check_full_prompt_substitution(records, task_name)
        task_warnings.extend(w)
        logger.info(f"  [6] Prompt substitution: {len(w)} warnings")

        # 7. Reference-present check (every task must carry a gold reference)
        w = check_reference_present(records, task_name)
        task_warnings.extend(w)
        logger.info(f"  [7] Reference present: {len(w)} warnings")

        # Determine pass/fail
        passed = len(task_errors) == 0
        total_variants = sum(len(r.get("variants", [])) for r in records)

        summary_table.append({
            "task": task_name,
            "instances": len(records),
            "variants": total_variants,
            "passed": passed,
            "warnings": len(task_warnings),
            "errors": len(task_errors),
        })

        report[task_name] = {
            "file": filepath,
            "n_instances": len(records),
            "total_variants": total_variants,
            "passed": passed,
            "n_warnings": len(task_warnings),
            "n_errors": len(task_errors),
            "warnings": task_warnings[:50],  # Cap at 50 for readability
            "errors": task_errors,
            "similarity_stats": sim_stats,
        }

    # Print summary table
    print("\n" + "=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)
    header = f"{'Task':<16} {'Instances':>10} {'Variants':>10} {'Passed':>8} {'Warnings':>10} {'Errors':>8}"
    print(header)
    print("-" * 90)

    for row in summary_table:
        status = "✓ PASS" if row["passed"] else "✗ FAIL"
        print(
            f"{row['task']:<16} "
            f"{row['instances']:>10} "
            f"{row['variants']:>10} "
            f"{status:>8} "
            f"{row['warnings']:>10} "
            f"{row['errors']:>8}"
        )

    print("=" * 90)

    # Save report
    report_path = os.path.join(data_dir, "gensens_validation_report.json")
    report["_summary"] = summary_table
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nValidation report saved to: {report_path}")

    return report


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="GenSens Dataset Validator")
    _parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to data directory (default: ../data relative to script)",
    )
    _parser.add_argument(
        "--expected-instances", type=int, default=200,
        help="Expected number of instances per task (default: 200; use 5 for smoke test)",
    )
    _parser.add_argument(
        "--expected-variants", type=int, default=8,
        help="Expected number of variants per instance (default: 8; use 4 for smoke test)",
    )
    _parser.add_argument(
        "--task", type=str, default=None,
        help="Validate only this task (default: all 4 tasks)",
    )
    _args = _parser.parse_args()

    project_root = Path(__file__).parent.parent
    _data_dir = _args.data_dir if _args.data_dir else str(project_root / "data")

    validate_dataset(
        _data_dir,
        expected_instances=_args.expected_instances,
        expected_variants=_args.expected_variants,
    )
