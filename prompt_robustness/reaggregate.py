#!/usr/bin/env python3
"""
reaggregate.py — recompute composite scores from scored_samples.csv (NO models).

This is the zero-GPU evaluation-iteration loop: the expensive H100 generation +
scoring is persisted once to results/scored_samples.csv (Layer C); here we
recompute PRI / ORI / IFI / Final_Score and the honest correlations purely
arithmetically. Tune the PRI weights and re-rank models in seconds, no GPU.

Usage:
    python reaggregate.py [--scored results/scored_samples.csv]
                          [--w-consistency 0.40 --w-quality 0.35 --w-faith 0.25]
"""

import argparse
import numpy as np
import pandas as pd

from src.csv_io import validate_scored_csv
from src import scores as S   # SINGLE SOURCE OF TRUTH — same formulas as evaluator.py


def reaggregate(df: pd.DataFrame, w_c: float, w_q: float, w_f: float) -> pd.DataFrame:
    """Recompute per-sample composites using the shared formulas in src/scores.py."""
    d = df.copy()
    d["PRI"] = d.apply(
        lambda r: S.compute_pri(r["sms"], r["cs"], r["faithfulness"], r["avg_length"],
                                w_c, w_q, w_f), axis=1)
    d["ORI"] = d.apply(lambda r: S.compute_ori(r["sms"], r["auc_e"], r["trd"], r["kpig"]), axis=1)
    d["IFI"] = d.apply(lambda r: S.compute_ifi(r["ppl_var"], r["bf"]), axis=1)
    d["Diagnostic_ORI"] = d.apply(
        lambda r: S.compute_diagnostic_ori(r["sms"], r["auc_e"], r["trd"], r["kpig"]), axis=1)
    d["Diagnostic_IFI"] = d.apply(
        lambda r: S.compute_diagnostic_ifi(r["ppl_var"], r["bf"]), axis=1)
    d["Diagnostic_PRI"] = d.apply(
        lambda r: S.compute_diagnostic_pri(r["Diagnostic_ORI"], r["Diagnostic_IFI"]), axis=1)
    d["Diagnosis"] = d.apply(
        lambda r: S.compute_dual_pillar_diagnosis(r["Diagnostic_ORI"], r["Diagnostic_IFI"]), axis=1)
    d["Final_Score"] = d.apply(lambda r: S.compute_final_static(r["PRI"], r["human_score"]), axis=1)
    d["iPRI"] = (d["PRI"] * d["cs"].clip(0, 1)).clip(0, 1)
    d["Consistency"] = d["sms"].clip(0, 1)
    return d


def _corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="results/scored_samples.csv")
    ap.add_argument("--w-consistency", type=float, default=0.40)
    ap.add_argument("--w-quality", type=float, default=0.35)
    ap.add_argument("--w-faith", type=float, default=0.25)
    ap.add_argument("--out", default="results/benchmark_reaggregated.csv")
    args = ap.parse_args()

    ok, issues = validate_scored_csv(args.scored)
    if not ok:
        print(f"⚠️  scored CSV issues: {issues}")

    df = pd.read_csv(args.scored)
    total = args.w_consistency + args.w_quality + args.w_faith
    if abs(total - 1.0) > 1e-6:
        print(f"⚠️  PRI weights sum to {total:.3f} (not 1.0)")

    d = reaggregate(df, args.w_consistency, args.w_quality, args.w_faith)

    agg = d.groupby("model").agg(
        PRI=("PRI", "mean"), ORI=("ORI", "mean"), IFI=("IFI", "mean"),
        Diagnostic_PRI=("Diagnostic_PRI", "mean"),
        Diagnostic_ORI=("Diagnostic_ORI", "mean"),
        Diagnostic_IFI=("Diagnostic_IFI", "mean"),
        Consistency=("Consistency", "mean"), CS=("cs", "mean"),
        Faithfulness=("faithfulness", "mean"), HS=("hs", "mean"),
        Human=("human_score", "mean"), Final=("Final_Score", "mean"),
        n=("PRI", "size"),
    ).round(3).sort_values("PRI", ascending=False)

    print(f"\nPRI weights: consistency={args.w_consistency} quality={args.w_quality} faith={args.w_faith}")
    print("\n=== Re-aggregated per-model scores (no GPU) ===")
    print(agg.to_string())

    print("\n=== Honest correlations vs Human (non-circular) ===")
    for col in ["PRI", "Diagnostic_PRI", "Consistency", "cs", "faithfulness"]:
        print(f"  {col:14s} vs Human : {_corr(d[col], d['human_score']):+}")
    print("  (Final_Score vs Human intentionally omitted — circular)")

    d.to_csv(args.out, index=False)
    print(f"\nWrote per-sample reaggregation → {args.out}")


if __name__ == "__main__":
    main()
