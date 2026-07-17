"""
run_pirc_ablations.py — LL-PIRC ablation sweep driver.

PUBLICATION_SPEC §10: the LL-PIRC method paper requires three ablations
to defend the design choices:

  1. ℓ* sweep             — sensitive-layer position
                            (default scan: scan_start, ℓ*-4, ℓ*-2, ℓ*, ℓ*+2,
                            ℓ*+4, last_layer)
  2. α sweep              — clamping strength
                            (default scan: 0.0, 0.25, 0.50, 0.75, 1.0)
  3. anchor-percentile sweep — fraction of anchor tokens
                            (default scan: 10, 20, 30, 50, 70)

For each (knob, value) pair the script:
  - launches `experiment_pirc.py` with the override CLI flag
  - copies the produced `pirc.json` / `eval_summary.json` into a per-run
    sub-directory (`results/ablations/<knob>=<value>/`)
  - extracts the headline metrics (mean ROUGE-L variance reduction,
    mean ROUGE-L quality change, mean worst-prompt ROUGE-L)
  - aggregates per-knob CSVs ready for paper-figure rendering

USAGE
-----
  # Standard ablation (run sequentially on the GPU box)
  python prompt_robustness/run_pirc_ablations.py \\
      --config prompt_robustness/config.yaml \\
      --results-dir results/ablations \\
      --baseline-results results

  # Restrict to one knob (e.g. α only)
  python prompt_robustness/run_pirc_ablations.py \\
      --config prompt_robustness/config.yaml \\
      --results-dir results/ablations \\
      --baseline-results results \\
      --knobs alpha \\
      --alpha-values 0.0 0.25 0.5 0.75 1.0

  # Dry-run: print the planned schedule without launching any experiment
  python prompt_robustness/run_pirc_ablations.py \\
      --config prompt_robustness/config.yaml \\
      --results-dir results/ablations \\
      --baseline-results results \\
      --dry-run

OUTPUT
------
  results/ablations/<knob>=<value>/pirc.json
  results/ablations/<knob>=<value>/eval_summary.json
  results/ablations/<knob>_ablation.csv
  results/ablations/ablation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("run_pirc_ablations")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_ALPHA_VALUES   = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_ELL_STAR_VALUES = []  # filled in dynamically from baseline ℓ* +/- offsets
DEFAULT_ANCHOR_PERCENTILES = [10.0, 20.0, 30.0, 50.0, 70.0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-run launcher
# ─────────────────────────────────────────────────────────────────────────────

def _launch_one(
    config_path: Path,
    cli_flag: str,
    value: Any,
    src_results_dir: Path,
    dst_results_dir: Path,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Invoke experiment_pirc.py with a single override flag.

    Strategy: the experiment loads `--config` AND reads/writes baseline.json
    and pirc.json from `experiment.results_dir` in the YAML. We do NOT mutate
    the YAML — instead we run the experiment in-place, then move the produced
    pirc.json + eval_summary.json into `dst_results_dir` so the next run
    can overwrite the source without clobbering this run's artefacts.
    """
    args = [
        sys.executable,
        str(Path(__file__).parent / "experiment_pirc.py"),
        "--config", str(config_path),
        cli_flag, str(value),
    ]
    if extra_args:
        args.extend(extra_args)
    logger.info("  launching: " + " ".join(args))
    t0 = time.time()
    rc = subprocess.call(args)
    elapsed = time.time() - t0
    if rc != 0:
        logger.warning(f"  experiment_pirc.py exited rc={rc} after {elapsed:.1f}s")
        return rc

    dst_results_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("pirc.json", "eval_summary.json"):
        src = src_results_dir / fname
        if src.exists():
            shutil.copy2(src, dst_results_dir / fname)
        else:
            logger.warning(f"  expected {src} not produced by experiment_pirc")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_headline(eval_path: Path, pirc_path: Path) -> Dict[str, Any]:
    """Return the headline metrics for one ablation run."""
    headline: Dict[str, Any] = {
        "variance_reduction": None,
        "rouge_change":       None,
        "rouge_change_pct":   None,
        "mean_ell_star":      None,
        "wilcoxon_var_p":     None,
        "rouge_dz":           None,
        "var_dz":             None,
        "worst_prompt_rouge": None,
        "n_articles":         None,
    }
    if not eval_path.exists():
        logger.warning(f"  no eval_summary.json at {eval_path}")
        return headline
    with open(eval_path) as f:
        es = json.load(f)
    headline.update({
        "variance_reduction": es.get("relative_var_reduction"),
        "rouge_change":       es.get("rouge_change"),
        "rouge_change_pct":   es.get("rouge_change_pct"),
        "mean_ell_star":      (es.get("ell_star_stats") or {}).get("mean"),
        "wilcoxon_var_p":     (es.get("wilcoxon_variance") or {}).get("p_value"),
        "rouge_dz":           es.get("rouge_effect_size_dz"),
        "var_dz":             es.get("variance_effect_size_dz"),
        "n_articles":         es.get("num_articles"),
    })

    # Worst-prompt ROUGE-L — derived from pirc.json per-article ROUGE arrays.
    if pirc_path.exists():
        with open(pirc_path) as f:
            pdoc = json.load(f)
        per_article_mins: List[float] = []
        for r in pdoc.get("results", []):
            scores = r.get("pirc_rouge_scores") or []
            if scores:
                per_article_mins.append(float(min(scores)))
        if per_article_mins:
            headline["worst_prompt_rouge"] = sum(per_article_mins) / len(per_article_mins)
    return headline


# ─────────────────────────────────────────────────────────────────────────────
# Per-knob sweep
# ─────────────────────────────────────────────────────────────────────────────

def _sweep_one_knob(
    knob: str,
    cli_flag: str,
    values: List[Any],
    config_path: Path,
    baseline_results_dir: Path,
    ablations_root: Path,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    """Sweep one knob over a list of values; return per-value metric rows."""
    rows: List[Dict[str, Any]] = []
    print(f"\n══ Sweep: {knob} over {values} ══")
    for v in values:
        dst = ablations_root / f"{knob}={v}"
        if dry_run:
            print(f"  [dry-run] would run {knob}={v} → {dst}")
            rows.append({knob: v, "dst": str(dst), "dry_run": True})
            continue
        rc = _launch_one(
            config_path=config_path,
            cli_flag=cli_flag,
            value=v,
            src_results_dir=baseline_results_dir,
            dst_results_dir=dst,
        )
        if rc != 0:
            rows.append({knob: v, "dst": str(dst), "failed": True})
            continue
        headline = _extract_headline(
            eval_path=dst / "eval_summary.json",
            pirc_path=dst / "pirc.json",
        )
        rows.append({knob: v, **headline, "dst": str(dst)})
        print(f"  {knob}={v} done  "
              f"var↓={(headline['variance_reduction'] or 0)*100:5.1f}%  "
              f"ΔROUGE={(headline['rouge_change'] or 0):+.4f}  "
              f"worst-prompt={headline['worst_prompt_rouge'] or 'n/a'}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV + markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _write_knob_csv(rows: List[Dict[str, Any]], knob: str, out_path: Path) -> None:
    if not rows:
        return
    fields = [knob, "variance_reduction", "rouge_change", "rouge_change_pct",
              "mean_ell_star", "wilcoxon_var_p", "rouge_dz", "var_dz",
              "worst_prompt_rouge", "n_articles", "dst"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _render_markdown_summary(per_knob: Dict[str, List[Dict[str, Any]]]) -> str:
    md: List[str] = ["# LL-PIRC Ablation Summary\n"]
    for knob, rows in per_knob.items():
        if not rows or all(r.get("dry_run") for r in rows):
            continue
        md.append(f"\n## Knob: `{knob}`\n")
        md.append(f"| {knob} | Variance reduction | ΔROUGE-L | Mean worst-prompt ROUGE-L | Wilcoxon-var p |")
        md.append("|---|---|---|---|---|")
        for r in rows:
            if r.get("failed"):
                md.append(f"| {r.get(knob)} | FAILED | — | — | — |")
                continue
            vr  = r.get("variance_reduction")
            rc  = r.get("rouge_change")
            wp  = r.get("worst_prompt_rouge")
            wp_s = "—" if wp is None else f"{wp:.4f}"
            vr_s = "—" if vr is None else f"{100*vr:.1f}%"
            rc_s = "—" if rc is None else f"{rc:+.4f}"
            pv = r.get("wilcoxon_var_p")
            pv_s = "—" if pv is None else f"{pv:.4f}"
            md.append(f"| {r.get(knob)} | {vr_s} | {rc_s} | {wp_s} | {pv_s} |")
    md.append("\n> Variance reduction = `1 - mean(var_pirc) / mean(var_baseline)`")
    md.append("> Mean worst-prompt ROUGE-L = `mean_a [min_k ROUGE-L(output_{a,k}, gold_a)]`")
    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to config.yaml (same one experiment_pirc.py uses)")
    p.add_argument("--results-dir", required=True,
                   help="Output root for ablation artefacts")
    p.add_argument("--baseline-results", required=True,
                   help="experiment_pirc results dir (the one config points to)")

    p.add_argument("--knobs", nargs="+",
                   choices=["alpha", "ell_star", "anchor_percentile"],
                   default=["alpha", "anchor_percentile"],
                   help="Which knobs to sweep (default: alpha + anchor_percentile; "
                        "include 'ell_star' explicitly to sweep the layer too)")

    p.add_argument("--alpha-values", nargs="+", type=float,
                   default=DEFAULT_ALPHA_VALUES,
                   help="alpha sweep values (default: 0 0.25 0.5 0.75 1.0)")
    p.add_argument("--ell-star-values", nargs="+", type=int,
                   default=None,
                   help="ℓ* sweep values. Required if 'ell_star' is in --knobs.")
    p.add_argument("--anchor-percentile-values", nargs="+", type=float,
                   default=DEFAULT_ANCHOR_PERCENTILES,
                   help="anchor percentile sweep values (default: 10 20 30 50 70)")

    p.add_argument("--dry-run", action="store_true",
                   help="Print schedule without launching any subprocess")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    config_path           = Path(args.config)
    ablations_root        = Path(args.results_dir)
    baseline_results_dir  = Path(args.baseline_results)

    if not config_path.exists():
        sys.exit(f"ERROR: config not found: {config_path}")
    if not baseline_results_dir.exists():
        sys.exit(f"ERROR: baseline-results dir not found: {baseline_results_dir}")

    ablations_root.mkdir(parents=True, exist_ok=True)

    per_knob: Dict[str, List[Dict[str, Any]]] = {}

    if "alpha" in args.knobs:
        rows = _sweep_one_knob(
            "alpha", "--alpha", args.alpha_values,
            config_path, baseline_results_dir, ablations_root, args.dry_run,
        )
        _write_knob_csv(rows, "alpha", ablations_root / "alpha_ablation.csv")
        per_knob["alpha"] = rows

    if "anchor_percentile" in args.knobs:
        rows = _sweep_one_knob(
            "anchor_percentile", "--anchor-percentile",
            args.anchor_percentile_values,
            config_path, baseline_results_dir, ablations_root, args.dry_run,
        )
        _write_knob_csv(rows, "anchor_percentile",
                        ablations_root / "anchor_percentile_ablation.csv")
        per_knob["anchor_percentile"] = rows

    if "ell_star" in args.knobs:
        if not args.ell_star_values:
            sys.exit("ERROR: --ell-star-values is required when 'ell_star' is in --knobs")
        rows = _sweep_one_knob(
            "ell_star", "--ell-star", args.ell_star_values,
            config_path, baseline_results_dir, ablations_root, args.dry_run,
        )
        _write_knob_csv(rows, "ell_star", ablations_root / "ell_star_ablation.csv")
        per_knob["ell_star"] = rows

    summary_md = ablations_root / "ablation_summary.md"
    with open(summary_md, "w") as f:
        f.write(_render_markdown_summary(per_knob))
    print(f"\n✓ wrote {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
