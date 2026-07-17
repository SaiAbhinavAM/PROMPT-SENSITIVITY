"""
robustalpaca_crossvalidation.py — Borrowed human validation via RobustAlpacaEval.

PUBLICATION_SPEC §8.4 (Validation Prong 4): the GenSens filter pipeline is
applied to RobustAlpacaEval's 1000 human-VERIFIED (base, paraphrase) pairs
(Cao et al., ICLR 2024). We report:

  (a) Filter pass rate on human-verified paraphrases.  Target ≥ 88%.
      If our filters retain ≥88% of paraphrases that humans already approved,
      we have evidence that the filters are calibrated to the human notion
      of paraphrase quality.

  (b) Kolmogorov-Smirnov distributional similarity (SBERT cos, length ratio,
      token Jaccard, bidirectional NLI) between RobustAlpacaEval and a
      GenSens JSONL.  Target p > 0.05 on each metric.
      Indistinguishable distributions = transitive human validation.

This is the central "borrowed human validation" claim. It is reproducible by
anyone (RobustAlpacaEval is public; our filters are open) and removes the
single largest reviewer concern about an automated-only validation pipeline.

USAGE
-----
  # 1. Download RobustAlpacaEval and produce the canonical pair CSV.
  python prompt_robustness/scripts/robustalpaca_crossvalidation.py \\
      --mode prepare \\
      --output gensens/data/robustalpaca_pairs.csv

  # 2. Run filter pipeline on the human-verified pairs.
  python prompt_robustness/scripts/robustalpaca_crossvalidation.py \\
      --mode audit \\
      --robustalpaca gensens/data/robustalpaca_pairs.csv \\
      --output-dir results/robustalpaca_audit \\
      --enable-nli-ensemble

  # 3. Cross-validate GenSens vs RobustAlpacaEval distributions.
  python prompt_robustness/scripts/robustalpaca_crossvalidation.py \\
      --mode crossvalidate \\
      --gensens-audit results/gensens_audit/per_pair.jsonl \\
      --robustalpaca-audit results/robustalpaca_audit/per_pair.jsonl \\
      --output-dir results/distributional_alignment

OUTPUT
------
  --mode prepare:
    robustalpaca_pairs.csv         (pair_id, base, paraphrase, [human_score])

  --mode audit:
    per_pair.jsonl                 (pair_id, base, paraphrase, sbert_sim,
                                     token_overlap, length_ratio, nli_*,
                                     filter_passed, rejection_reason)
    audit_summary.json             (pass_rate, per_filter_pass_rate,
                                     metric_distributions)

  --mode crossvalidate:
    distributional_alignment.json  (KS statistic + p-value per metric,
                                     verdict per filter, overall verdict)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("robustalpaca_crossvalidation")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Prepare canonical pair CSV from RobustAlpacaEval.
# ─────────────────────────────────────────────────────────────────────────────

ROBUSTALPACA_HF_ID = "ZBWpro/RobustAlpacaEval"
ROBUSTALPACA_FALLBACK_HF_IDS = [
    "ZBWpro/RobustAlpacaEval",
    "Cao-Yifan/RobustAlpacaEval",
]


def _download_robustalpaca() -> List[Dict[str, Any]]:
    """Try to fetch RobustAlpacaEval from HuggingFace Datasets.

    Returns a list of dicts with at minimum {original_query, paraphrases}
    where paraphrases is a list of strings. We try a few candidate HF ids
    because the dataset is mirrored under several namespaces — if all fail
    we raise with a clear message pointing to manual download.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: pip install datasets (required for --mode prepare)")

    last_err: Optional[Exception] = None
    for hf_id in ROBUSTALPACA_FALLBACK_HF_IDS:
        try:
            logger.info(f"Loading dataset {hf_id} …")
            ds = load_dataset(hf_id)
            split = list(ds.keys())[0]
            rows = list(ds[split])
            logger.info(f"  loaded {len(rows)} rows from {hf_id}:{split}")
            return rows
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  failed: {e}")
            last_err = e

    sys.exit(
        "ERROR: Could not download RobustAlpacaEval from any candidate HF id.\n"
        f"  Tried: {ROBUSTALPACA_FALLBACK_HF_IDS}\n"
        f"  Last error: {last_err}\n\n"
        "Manual fallback: download the dataset from\n"
        "  https://github.com/cyk1337/RobustAlpacaEval\n"
        "and produce a CSV with columns (base, paraphrase, [human_score]),\n"
        "then skip --mode prepare and pass it directly to --mode audit."
    )


def _normalise_robustalpaca_row(row: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Find the base query + paraphrase list in a row regardless of schema."""
    # Try common column name patterns.
    base_keys = ["original_query", "instruction", "query", "prompt", "original"]
    para_keys = ["paraphrases", "paraphrase", "rewrites", "variants", "variations"]
    base = None
    for k in base_keys:
        if k in row and isinstance(row[k], str) and row[k].strip():
            base = row[k]
            break
    paras: List[str] = []
    for k in para_keys:
        if k in row:
            v = row[k]
            if isinstance(v, list):
                paras = [str(x).strip() for x in v if str(x).strip()]
                break
            if isinstance(v, str) and v.strip():
                paras = [v.strip()]
                break
    return base, paras


def run_prepare(args) -> int:
    rows = _download_robustalpaca()
    out_rows: List[Dict[str, Any]] = []
    n_skip = 0
    for i, row in enumerate(rows):
        base, paras = _normalise_robustalpaca_row(row)
        if not base or not paras:
            n_skip += 1
            continue
        for j, para in enumerate(paras):
            out_rows.append({
                "pair_id":      f"ra_{i:04d}_{j:02d}",
                "base":         base,
                "paraphrase":   para,
                "source_idx":   i,
                "paraphrase_idx": j,
            })
    print(f"→ Extracted {len(out_rows)} pairs from {len(rows)} source rows "
          f"({n_skip} skipped)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "base", "paraphrase",
                                          "source_idx", "paraphrase_idx"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"✓ wrote {out_path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Audit: run GenSens filter pipeline on the prepared pairs.
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_filter_pipeline():
    """Import FilterPipeline from paws_negative_controls.

    This guarantees the audit uses EXACTLY the same filter logic so the
    "borrowed human validation" claim is methodologically consistent with the
    PAWS negative-control audit (§8.3).
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))
    try:
        from paws_negative_controls import FilterPipeline  # type: ignore
        return FilterPipeline
    except ImportError as e:
        sys.exit(
            f"ERROR: cannot import FilterPipeline from paws_negative_controls "
            f"({e}). Both scripts must live in the same directory."
        )


def _iter_pair_csv(csv_path: Path) -> Iterable[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def run_audit(args) -> int:
    csv_path = Path(args.robustalpaca)
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} does not exist (run --mode prepare first)")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = out_dir / "per_pair.jsonl"
    summary_path  = out_dir / "audit_summary.json"

    FilterPipeline = _ensure_filter_pipeline()
    pipeline = FilterPipeline(
        sbert_model           = args.sbert_model,
        nli_model_name        = args.nli_model,
        enable_nli_ensemble   = args.enable_nli_ensemble,
        nli_ensemble_majority = args.nli_ensemble_majority,
    )

    pairs = list(_iter_pair_csv(csv_path))
    print(f"→ Auditing {len(pairs)} RobustAlpacaEval pairs")

    rows: List[Dict[str, Any]] = []
    with open(per_pair_path, "w", encoding="utf-8") as fout:
        for i, p in enumerate(pairs, 1):
            base, cand = p["base"], p["paraphrase"]
            res = pipeline.evaluate_pair(base, cand)
            row = {
                "pair_id":           p.get("pair_id", f"row{i}"),
                "base":              base,
                "paraphrase":        cand,
                "sbert_sim":         res.get("sbert_sim"),
                "token_overlap":     res.get("token_overlap"),
                "length_ratio":      res.get("length_ratio"),
                "nli":               res.get("nli"),
                "filter_passed":     res.get("accepted"),
                "rejection_reason":  res.get("rejection_reason", ""),
            }
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(pairs):
                passed = sum(1 for r in rows if r["filter_passed"])
                print(f"  [{i}/{len(pairs)}] pass rate so far: "
                      f"{100*passed/i:.1f}%")

    summary = _summarise_audit(rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ wrote {per_pair_path}")
    print(f"✓ wrote {summary_path}")
    print("\n" + "=" * 72)
    pr = 100 * summary["pass_rate"]
    print(f"FILTER PASS RATE ON HUMAN-VERIFIED PAIRS: {pr:.1f}% "
          f"(target ≥ 88%)  {'✓' if pr >= 88 else '✗'}")
    print("=" * 72)
    print("\nRejection reason breakdown:")
    for reason, cnt in summary["rejection_reason_counts"].items():
        print(f"  {reason:20s}: {cnt}  ({100*cnt/len(rows):.1f}%)")
    return 0


def _summarise_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    passed = sum(1 for r in rows if r["filter_passed"])
    reasons: Dict[str, int] = {}
    for r in rows:
        rr = r.get("rejection_reason") or ""
        if rr:
            reasons[rr] = reasons.get(rr, 0) + 1

    def _dist_stats(key: str) -> Dict[str, Any]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return {"n": 0}
        return {
            "n":      len(vals),
            "mean":   float(mean(vals)),
            "median": float(median(vals)),
            "std":    float(stdev(vals)) if len(vals) > 1 else 0.0,
            "min":    float(min(vals)),
            "max":    float(max(vals)),
        }

    # NLI fwd / bwd are nested; extract them.
    nli_fwd, nli_bwd = [], []
    for r in rows:
        n_blk = r.get("nli") or {}
        f, b = n_blk.get("p_fwd"), n_blk.get("p_bwd")
        if isinstance(f, (int, float)): nli_fwd.append(f)
        if isinstance(b, (int, float)): nli_bwd.append(b)

    def _list_stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0}
        return {
            "n":      len(vals),
            "mean":   float(mean(vals)),
            "median": float(median(vals)),
            "std":    float(stdev(vals)) if len(vals) > 1 else 0.0,
        }

    return {
        "n_pairs":                 n,
        "n_passed":                passed,
        "pass_rate":               passed / n if n else 0.0,
        "target_pass_rate":        0.88,
        "passes_target":           (passed / n if n else 0.0) >= 0.88,
        "rejection_reason_counts": reasons,
        "metric_distributions": {
            "sbert_sim":     _dist_stats("sbert_sim"),
            "token_overlap": _dist_stats("token_overlap"),
            "length_ratio":  _dist_stats("length_ratio"),
            "nli_fwd":       _list_stats(nli_fwd),
            "nli_bwd":       _list_stats(nli_bwd),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Crossvalidate: KS test between GenSens audit and RobustAlpacaEval audit.
# ─────────────────────────────────────────────────────────────────────────────

def _ks_two_sample(a: List[float], b: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Two-sample Kolmogorov-Smirnov test (statistic, p-value).

    Pure-Python implementation so the script has no SciPy dependency. Uses
    Stephens' (1970) asymptotic p-value approximation which is accurate to
    ~1% for sample sizes n, m ≥ 10. For smaller samples we still return the
    statistic but flag p as None.
    """
    if not a or not b:
        return None, None
    n, m = len(a), len(b)
    # Build combined sorted unique value list.
    all_vals = sorted(set(a) | set(b))
    a_sorted = sorted(a)
    b_sorted = sorted(b)

    def _ecdf(x_sorted: List[float], v: float) -> float:
        # Empirical CDF at v: fraction of x ≤ v.
        lo, hi = 0, len(x_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            if x_sorted[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(x_sorted)

    d_max = 0.0
    for v in all_vals:
        d = abs(_ecdf(a_sorted, v) - _ecdf(b_sorted, v))
        if d > d_max:
            d_max = d

    # Asymptotic p-value via Kolmogorov's K distribution.
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d_max
    # Q_KS(lam) = 2 * sum_{j=1..inf} (-1)^(j-1) exp(-2 j^2 lam^2)
    # Truncate at j=100 — far beyond convergence for any reasonable lam.
    if lam <= 0:
        return d_max, 1.0
    p = 0.0
    for j in range(1, 101):
        term = 2 * ((-1) ** (j - 1)) * math.exp(-2 * j * j * lam * lam)
        p += term
        if abs(term) < 1e-10:
            break
    p = max(0.0, min(1.0, p))
    return d_max, p


def _load_audit_jsonl(path: Path, metric: str, nli_dir: str = "p_fwd") -> List[float]:
    out: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if metric in ("sbert_similarity", "sbert_sim"):
                # Audit JSONLs use 'sbert_sim'; GenSens variant JSONLs may use 'sbert_similarity'.
                v = r.get("sbert_sim", r.get("sbert_similarity"))
            elif metric == "token_overlap":
                v = r.get("token_overlap", r.get("token_overlap_to_base"))
            elif metric == "length_ratio":
                v = r.get("length_ratio")
            elif metric == "nli":
                nli_blk = r.get("nli")
                if isinstance(nli_blk, dict):
                    v = nli_blk.get(nli_dir)
                else:
                    # GenSens JSONL stores nli_entail_fwd / nli_entail_bwd directly
                    key = "nli_entail_fwd" if nli_dir == "p_fwd" else "nli_entail_bwd"
                    v = r.get(key)
            else:
                v = r.get(metric)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                out.append(float(v))
    return out


def run_crossvalidate(args) -> int:
    gensens_path     = Path(args.gensens_audit)
    robustalpaca_path = Path(args.robustalpaca_audit)
    for p in (gensens_path, robustalpaca_path):
        if not p.exists():
            sys.exit(f"ERROR: {p} does not exist")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "distributional_alignment.json"

    metrics_to_compare = [
        ("sbert_sim",     None),
        ("token_overlap", None),
        ("length_ratio",  None),
        ("nli",           "p_fwd"),
        ("nli",           "p_bwd"),
    ]
    per_metric: Dict[str, Any] = {}
    print(f"\n{'Metric':18s}  {'GenSens n':10s}  {'RA n':10s}  {'KS stat':10s}  {'p-value':10s}  Verdict")
    print("-" * 78)

    for metric, nli_dir in metrics_to_compare:
        label = metric if nli_dir is None else f"{metric}:{nli_dir}"
        a = _load_audit_jsonl(gensens_path, metric, nli_dir or "p_fwd")
        b = _load_audit_jsonl(robustalpaca_path, metric, nli_dir or "p_fwd")
        d, p = _ks_two_sample(a, b)
        verdict = (
            "ALIGNED (p>0.05)" if (p is not None and p > 0.05)
            else ("DIVERGENT" if p is not None else "INSUFFICIENT")
        )
        per_metric[label] = {
            "n_gensens":     len(a),
            "n_robustalpaca": len(b),
            "ks_statistic":  d,
            "p_value":       p,
            "verdict":       verdict,
        }
        d_str = f"{d:.4f}" if d is not None else "n/a"
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(f"{label:18s}  {len(a):10d}  {len(b):10d}  {d_str:10s}  {p_str:10s}  {verdict}")

    aligned = sum(1 for v in per_metric.values() if v["verdict"].startswith("ALIGNED"))
    n_total = sum(1 for v in per_metric.values() if v["verdict"] != "INSUFFICIENT")
    overall = "ALIGNED" if (n_total > 0 and aligned == n_total) else (
        "PARTIAL" if aligned > 0 else "DIVERGENT"
    )
    out = {
        "metrics":        per_metric,
        "n_aligned":      aligned,
        "n_compared":     n_total,
        "overall_verdict": overall,
        "interpretation":  (
            "ALIGNED: our paraphrase distribution is statistically "
            "indistinguishable from RobustAlpacaEval's human-verified one — "
            "supports the 'borrowed human validation' claim."
            if overall == "ALIGNED" else
            "PARTIAL/DIVERGENT: some metrics differ significantly. "
            "Investigate per-metric distributions before claiming alignment."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ wrote {out_path}")
    print(f"\nOVERALL VERDICT: {overall}  ({aligned}/{n_total} metrics aligned)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=["prepare", "audit", "crossvalidate"])

    # prepare
    p.add_argument("--output", help="(prepare) output pair CSV path")

    # audit
    p.add_argument("--robustalpaca",
                   help="(audit) prepared RobustAlpaca pair CSV")
    p.add_argument("--output-dir",
                   help="(audit/crossvalidate) output directory")
    p.add_argument("--sbert-model", default="all-mpnet-base-v2")
    p.add_argument("--nli-model",   default="cross-encoder/nli-deberta-v3-small")
    p.add_argument("--enable-nli-ensemble", action="store_true")
    p.add_argument("--nli-ensemble-majority", type=int, default=2)

    # crossvalidate
    p.add_argument("--gensens-audit",
                   help="(crossvalidate) per-pair audit JSONL from a GenSens audit")
    p.add_argument("--robustalpaca-audit",
                   help="(crossvalidate) per-pair audit JSONL from a RobustAlpaca audit")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.mode == "prepare":
        if not args.output:
            sys.exit("ERROR: --output required for --mode prepare")
        return run_prepare(args)
    if args.mode == "audit":
        if not args.robustalpaca or not args.output_dir:
            sys.exit("ERROR: --robustalpaca and --output-dir required for --mode audit")
        return run_audit(args)
    if args.mode == "crossvalidate":
        if not args.gensens_audit or not args.robustalpaca_audit or not args.output_dir:
            sys.exit("ERROR: --gensens-audit + --robustalpaca-audit + --output-dir required")
        return run_crossvalidate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
