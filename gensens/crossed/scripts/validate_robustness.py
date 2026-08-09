#!/usr/bin/env python3
"""
validate_robustness.py — post-hoc validation / cross-checks for a completed
crossed-design run. Pure CPU (numpy/scipy), operates on cell_metrics_scored.jsonl
and the seed dataset; no GPU, no re-inference.

Four checks (method.md 2026-08-09 validation entry):
  1. LENGTH CONFOUND    — is Sensitivity just an output-length effect? Reports
     corr(Sensitivity, mean_output_len) and the dimension-ranking Spearman before
     vs after regressing out length. High ranking-Spearman => not a length artifact.
  2. SPLIT-HALF         — does the dimension ranking replicate within the study?
     Many random article half-splits; mean Spearman of the two independent rankings.
  3. PERMUTATION TEST   — assumption-free significance of the dimension effect at
     the SEED level (shuffle dimension labels among seeds; no normality assumption).
  4. PARAPHRASE EQUIV.  — are the prompt paraphrases truly meaning-equivalent?
     Dataset-wide SBERT similarity + bidirectional NLI entailment + nli_passed rate
     + token overlap (want: high entailment, LOW token overlap).
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("validate_robustness")


def _dim_means(cells, key="sensitivity"):
    d = defaultdict(list)
    for c in cells:
        d[c["dimension"]].append(c[key])
    return {k: float(np.mean(v)) for k, v in d.items()}


def check_length_confound(cells):
    sens = np.array([c["sensitivity"] for c in cells], float)
    length = np.array([c["mean_output_len"] for c in cells], float)
    # regress Sensitivity on length, residualize, re-rank dimensions
    A = np.column_stack([np.ones(len(length)), length])
    beta, *_ = np.linalg.lstsq(A, sens, rcond=None)
    resid = sens - A @ beta
    for c, r in zip(cells, resid):
        c["_sens_len_resid"] = float(r)
    dims = sorted(set(c["dimension"] for c in cells))
    raw = _dim_means(cells, "sensitivity")
    res = _dim_means(cells, "_sens_len_resid")
    rho = float(spearmanr([raw[d] for d in dims], [res[d] for d in dims]).statistic)
    return {
        "pearson_sensitivity_length": float(pearsonr(sens, length)[0]),
        "spearman_sensitivity_length": float(spearmanr(sens, length).statistic),
        "dim_ranking_spearman_raw_vs_lengthresid": rho,
        "verdict": "length is NOT the driver" if rho > 0.9 else "length materially changes ranking — inspect",
    }


def check_split_half(cells, n_splits=300, seed=42):
    articles = sorted(set(c["article_id"] for c in cells))
    dims = sorted(set(c["dimension"] for c in cells))
    rng = np.random.default_rng(seed)
    half = len(articles) // 2
    cors = []
    for _ in range(n_splits):
        p = rng.permutation(articles)
        h1, h2 = set(p[:half]), set(p[half:])
        r1 = _dim_means([c for c in cells if c["article_id"] in h1])
        r2 = _dim_means([c for c in cells if c["article_id"] in h2])
        common_d = [d for d in dims if d in r1 and d in r2]
        if len(common_d) >= 3:
            cors.append(spearmanr([r1[d] for d in common_d], [r2[d] for d in common_d]).statistic)
    cors = np.array(cors)
    return {
        "n_splits": int(len(cors)),
        "mean_spearman": float(cors.mean()),
        "median_spearman": float(np.median(cors)),
        "ci90": [float(np.percentile(cors, 5)), float(np.percentile(cors, 95))],
        "frac_above_0.7": float(np.mean(cors > 0.7)),
        "verdict": "ranking replicates within-study" if cors.mean() > 0.7 else "ranking unstable across halves",
    }


def check_permutation(cells, n_perm=20000, seed=0):
    seed_vals = defaultdict(list)
    seed_dim = {}
    for c in cells:
        seed_vals[c["seed_id"]].append(c["sensitivity"])
        seed_dim[c["seed_id"]] = c["dimension"]
    seeds = sorted(seed_vals)
    sm = np.array([np.mean(seed_vals[s]) for s in seeds])
    labels = np.array([seed_dim[s] for s in seeds])

    def eta2(vals, labs):
        grand = vals.mean()
        sst = ((vals - grand) ** 2).sum()
        if sst <= 0:
            return float("nan")
        ssb = sum((labs == u).sum() * (vals[labs == u].mean() - grand) ** 2 for u in np.unique(labs))
        return float(ssb / sst)

    obs = eta2(sm, labels)
    rng = np.random.default_rng(seed)
    null = np.array([eta2(sm, rng.permutation(labels)) for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {
        "unit": "seed", "n_seeds": int(len(seeds)), "n_perm": n_perm,
        "observed_eta2": obs, "null_mean_eta2": float(null.mean()),
        "null_p95_eta2": float(np.percentile(null, 95)), "perm_p_value": p,
        "verdict": "SIGNIFICANT (assumption-free)" if p < 0.05 else "not significant (assumption-free)",
    }


def check_paraphrase_equivalence(seed_file=None):
    seeds = common.read_jsonl(seed_file or common.SEED_FILE_DEFAULT)
    if not seeds:
        return {"error": "seed dataset not found"}
    sim, ef, eb, tok, passed = [], [], [], [], []
    for s in seeds:
        for v in s.get("variants", []):
            for arr, k in ((sim, "sbert_similarity"), (ef, "nli_entail_fwd"),
                           (eb, "nli_entail_bwd"), (tok, "token_overlap_to_base")):
                if v.get(k) is not None:
                    arr.append(v[k])
            passed.append(v.get("nli_passed"))

    def stat(a):
        a = np.array(a, float)
        return {"mean": float(a.mean()), "min": float(a.min())} if len(a) else None

    return {
        "n_paraphrases": sum(len(s.get("variants", [])) for s in seeds),
        "sbert_similarity": stat(sim),
        "nli_entail_forward": stat(ef),
        "nli_entail_backward": stat(eb),
        "token_overlap_to_base": stat(tok),
        "nli_passed_rate": float(np.mean([p is True for p in passed])) if passed else None,
        "verdict": ("equivalent meaning (high bidirectional NLI) + different wording (low token overlap)"
                    if ef and np.mean(ef) > 0.9 and tok and np.mean(tok) < 0.6 else "inspect equivalence"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--seed_file", default=None)
    args = ap.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT
    cells = common.read_jsonl(results_dir / "cell_metrics_scored.jsonl")
    if not cells:
        raise FileNotFoundError(f"No cell_metrics_scored.jsonl in {results_dir}. Run aggregate_scores.py first.")
    log.info(f"Loaded {len(cells)} scored cells.")

    out = {
        "n_cells": len(cells),
        "length_confound": check_length_confound(cells),
        "split_half": check_split_half(cells),
        "permutation": check_permutation(cells),
        "paraphrase_equivalence": check_paraphrase_equivalence(args.seed_file),
    }
    (results_dir / "validation.json").write_text(json.dumps(out, indent=2))
    log.info(f"Wrote {results_dir / 'validation.json'}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
