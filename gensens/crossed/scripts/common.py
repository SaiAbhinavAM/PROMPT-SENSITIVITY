"""
common.py — shared paths, IO, and stats helpers for the crossed-design
GenSens summarization prompt-sensitivity benchmark.

Design note (why this module exists separately from gensens/scripts and
prompt_robustness/src): the crossed design (every seed x every article,
6 variants/cell) is a different experimental grid than the per-instance
1:1 pairing used elsewhere in the repo, and it deliberately separates
SENSITIVITY (spread across the 6 variant outputs in a cell) from QUALITY
(mean across those same 6 outputs) — see method.md 2026-07-06 entry.
Nothing here imports GPU-heavy libraries at module load time; those are
imported lazily inside the functions that need them so this file (and the
pure-aggregation scripts that import it) stays importable on a laptop.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
CROSSED_DIR = SCRIPTS_DIR.parent
GENSENS_DIR = CROSSED_DIR.parent

SEED_FILE_DEFAULT = GENSENS_DIR / "data" / "gensens_summ_50seed_10para.jsonl"
RESULTS_DIR_DEFAULT = CROSSED_DIR / "results"
LOGS_DIR_DEFAULT = CROSSED_DIR / "logs"

N_SEEDS_EXPECTED = 50
# Balanced 5-axis × 2 = 10 paraphrases/seed (method.md 2026-08-22):
# 1 base (variant_idx=-1) + 10 paraphrases (0..9).
N_VARIANTS_PER_SEED = 11
MIN_VARIANTS_PER_SEED = 4  # floor below which a seed is unusable, not just short

# KNOWN DATA-QUALITY FLAW (see method.md 2026-07-06 entry / gensens/data/fix_changelog.json):
# seed B12 has only 4 total variants (1 base + 3 paraphrases; a 4th paraphrase was
# flagged "oversim_replaced" upstream and never regenerated, unresolved=true). The
# crossed design tolerates this — it records the TRUE per-seed variant count rather
# than padding/faking a 6th output — but every other seed is expected at N_VARIANTS_PER_SEED.
# Re-run gensens/scripts/regenerate_flagged.py to close this gap before a publication run.


# ─────────────────────────────────────────────────────────────
# JSONL IO
# ─────────────────────────────────────────────────────────────
def read_jsonl(path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path) -> Iterator[Dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# Seed-prompt loading + validation
# ─────────────────────────────────────────────────────────────
def load_seed_prompts(path=None) -> List[Dict]:
    """Load the 50-seed x 5-paraphrase dataset and normalize each seed to
    exactly N_VARIANTS_PER_SEED (6) prompts: variant_idx -1 = base (built
    from base_prompt, which already carries the {{article}} placeholder),
    variant_idx 0..4 = the 5 paraphrases (full_prompt, also carrying the
    placeholder).

    Raises ValueError if the file is missing or doesn't have exactly 50
    seeds — the crossed design is only valid over the full seed pool.
    """
    path = Path(path) if path else SEED_FILE_DEFAULT
    if not path.exists():
        raise FileNotFoundError(
            f"Seed prompt file not found: {path}. "
            f"Expected the GenSens 50-seed/5-paraphrase dataset."
        )
    seeds = read_jsonl(path)
    if len(seeds) != N_SEEDS_EXPECTED:
        raise ValueError(
            f"Expected exactly {N_SEEDS_EXPECTED} seeds in {path}, found {len(seeds)}."
        )

    normalized = []
    for seed in seeds:
        seed_id = seed["prompt_id"]
        pool = seed["pool"]
        dimension = seed["dimension"]
        base_prompt = seed.get("base_prompt")
        if not base_prompt or "{{article}}" not in base_prompt:
            raise ValueError(f"Seed {seed_id}: base_prompt missing or has no {{{{article}}}} placeholder.")

        variants = [{
            "variant_idx": -1,
            "text": seed["base_text"],
            "full_prompt": base_prompt,
            "strategy": "base",
            "strategy_family": "base",
        }]
        paraphrases = seed.get("variants", [])
        for v in paraphrases:
            fp = v.get("full_prompt")
            if not fp or "{{article}}" not in fp:
                raise ValueError(
                    f"Seed {seed_id} variant {v.get('variant_idx')}: full_prompt missing "
                    f"or has no {{{{article}}}} placeholder."
                )
            variants.append({
                "variant_idx": v["variant_idx"],
                "text": v["paraphrased_text"],
                "full_prompt": fp,
                "strategy": v.get("strategy", "unknown"),
                # Axis label for per-axis sensitivity analysis (method.md 2026-08-22)
                "strategy_family": v.get("strategy_family", "unknown"),
            })

        if len(variants) < MIN_VARIANTS_PER_SEED:
            raise ValueError(
                f"Seed {seed_id}: only {len(variants)} total variants, below the "
                f"floor of {MIN_VARIANTS_PER_SEED} — this seed needs to be repaired "
                f"(see gensens/scripts/regenerate_flagged.py) before it can be used."
            )
        if len(variants) != N_VARIANTS_PER_SEED:
            logging.getLogger("common").warning(
                f"Seed {seed_id}: has {len(variants)} variants, expected {N_VARIANTS_PER_SEED} "
                f"(known upstream gap — see fix_changelog.json). Cells for this seed will use "
                f"n_variants={len(variants)} instead of {N_VARIANTS_PER_SEED}."
            )

        normalized.append({
            "seed_id": seed_id,
            "pool": pool,
            "dimension": dimension,
            "variants": variants,
        })
    return normalized


# ─────────────────────────────────────────────────────────────
# Stratified article sampling (CNN/DailyMail test split)
# ─────────────────────────────────────────────────────────────
def sample_articles(n_articles: int = 100, seed: int = 42, cache_path=None) -> List[Dict]:
    """Stratified sample of CNN/DailyMail test articles by word-count bin
    (<400 / 400-800 / >800 words), seed=42. Cached to `cache_path` (jsonl)
    so the expensive HF `datasets` load + sampling only happens once and
    the crossed run is reproducible/resumable across restarts.

    Each returned dict: {"article_id", "text", "gold_summary", "word_count"}.
    """
    cache_path = Path(cache_path) if cache_path else RESULTS_DIR_DEFAULT / "articles_sample.jsonl"
    cached = read_jsonl(cache_path)
    if len(cached) == n_articles:
        return cached
    if cached:
        raise ValueError(
            f"Cached article sample at {cache_path} has {len(cached)} articles, "
            f"expected {n_articles}. Delete the cache file to resample, or pass "
            f"the matching --n_articles."
        )

    from datasets import load_dataset  # lazy: heavy, network-fetching

    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

    bins = {"short": [], "medium": [], "long": []}
    for i, row in enumerate(ds):
        wc = len(row["article"].split())
        if wc < 400:
            bins["short"].append(i)
        elif wc <= 800:
            bins["medium"].append(i)
        else:
            bins["long"].append(i)

    rng = random.Random(seed)
    for b in bins.values():
        rng.shuffle(b)

    total = sum(len(v) for v in bins.values())
    per_bin_target = {}
    remaining = n_articles
    bin_names = list(bins.keys())
    for j, name in enumerate(bin_names):
        if j == len(bin_names) - 1:
            per_bin_target[name] = remaining
        else:
            share = round(n_articles * len(bins[name]) / total) if total else 0
            share = min(share, len(bins[name]), remaining)
            per_bin_target[name] = share
            remaining -= share

    selected_indices = []
    for name in bin_names:
        take = min(per_bin_target[name], len(bins[name]))
        selected_indices.extend(bins[name][:take])

    # Top up if stratified rounding left us short (small bins etc.)
    if len(selected_indices) < n_articles:
        used = set(selected_indices)
        pool = [i for b in bins.values() for i in b if i not in used]
        rng.shuffle(pool)
        selected_indices.extend(pool[: n_articles - len(selected_indices)])

    selected_indices = selected_indices[:n_articles]
    rng.shuffle(selected_indices)  # de-correlate bin order from article_id order

    articles = []
    for rank, idx in enumerate(selected_indices):
        row = ds[idx]
        articles.append({
            "article_id": f"art_{rank:04d}",
            "text": row["article"],
            "gold_summary": row["highlights"],
            "word_count": len(row["article"].split()),
        })

    write_jsonl(cache_path, articles)
    return articles


# ─────────────────────────────────────────────────────────────
# Stats helpers (pure numpy/scipy — no GPU deps)
# ─────────────────────────────────────────────────────────────
def bootstrap_ci(values: List[float], n_boot: int = 10000, seed: int = 42, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean. Returns (low, high)."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boot_means[i] = sample.mean()
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(low), float(high))


def cohens_dz(diffs: List[float]) -> float:
    """Paired Cohen's dz = mean(diff) / std(diff, ddof=1)."""
    diffs = np.asarray([d for d in diffs if d is not None], dtype=float)
    if len(diffs) < 2:
        return float("nan")
    sd = diffs.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diffs.mean() / sd)


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial effect size from Mann-Whitney U (r = 1 - 2U/(n1*n2))."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def min_max_normalize(values: List[Optional[float]]) -> List[Optional[float]]:
    """Min-max normalize to [0,1] using the empirical range of non-None
    values. None values (not-computed metrics) pass through unchanged so
    downstream weight-renormalization can detect and exclude them.

    WARNING: min-max is dominated by a single extreme value on heavy-tailed
    metrics. On the crossed-design spread metrics (cs_var, faith_var,
    pc_stab_var) a lone outlier cell set the max and collapsed 70–94% of
    cells to ≈0, so the nominal composite weights did not reflect actual
    contribution (method.md 2026-07-06 fix). Prefer rank_normalize for the
    Sensitivity composite; min_max is kept for reference / ablation."""
    present = [v for v in values if v is not None]
    if not present:
        return list(values)
    lo, hi = min(present), max(present)
    if hi - lo < 1e-12:
        return [0.0 if v is not None else None for v in values]
    return [((v - lo) / (hi - lo)) if v is not None else None for v in values]


def rank_normalize(values: List[Optional[float]]) -> List[Optional[float]]:
    """Rank/percentile normalize non-None values to [0, 1] (outlier-robust).

    Each value is mapped to its average-rank percentile across the non-None
    population: the smallest → ~0, the largest → 1, ties share the mean of
    their ranks. Unlike min-max this is invariant to the magnitude of extreme
    values, so a single heavy tail cannot compress the rest of the
    distribution — every spread metric then contributes its full intended
    dynamic range to the weighted composite. None passes through unchanged.

    With n present values, percentiles use (rank - 1) / (n - 1) so the min
    maps exactly to 0.0 and the max to 1.0 (n == 1 → 0.0).
    """
    present_idx = [i for i, v in enumerate(values) if v is not None]
    present = [values[i] for i in present_idx]
    n = len(present)
    if n == 0:
        return list(values)
    if n == 1:
        out = list(values)
        out[present_idx[0]] = 0.0
        return out

    order = sorted(range(n), key=lambda k: present[k])
    # Average-rank (1-based) with tie handling.
    avg_rank = [0.0] * n
    k = 0
    while k < n:
        j = k
        while j + 1 < n and present[order[j + 1]] == present[order[k]]:
            j += 1
        mean_rank = (k + j) / 2.0 + 1.0  # 1-based average of the tied block
        for t in range(k, j + 1):
            avg_rank[order[t]] = mean_rank
        k = j + 1

    out = list(values)
    for local_i, global_i in enumerate(present_idx):
        out[global_i] = (avg_rank[local_i] - 1.0) / (n - 1)
    return out


def coeff_of_variation(std: float, mean: float, eps: float = 1e-6, cap: float = 5.0) -> float:
    """Level-normalized spread = std / mean (a.k.a. coefficient of variation),
    capped and mean-guarded. Used to de-confound the spread of a bounded quality
    score (cs, faith) from its LEVEL: raw variance of a [0,1] score is
    mechanically tied to the mean (it is squeezed toward 0 as the mean nears a
    bound), so cells that happen to score higher/lower look artificially more/less
    'sensitive'. Dividing by the mean removes that level dependence and makes
    cs/faith spread consistent with ppl_var, which already used std/mean.
    Returns 0.0 when the mean is ~0 (no meaningful relative spread)."""
    if mean is None or std is None:
        return None
    if mean <= eps:
        return 0.0
    return float(min(std / mean, cap))


def ols_residual(y: List[float], *predictors: List[float]) -> np.ndarray:
    """Residual of y after an OLS fit on the given predictor columns (intercept
    added automatically) — pure numpy, no statsmodels. Used to strip the quality
    covariates (cs_mean, faith_mean) out of Sensitivity so the residual is a
    'pure spread' signal not confounded by 'less stable prompts are also slightly
    worse'. Returns y minus its fitted value (mean-zero by construction)."""
    y = np.asarray(y, dtype=float)
    cols = [np.ones(len(y))] + [np.asarray(p, dtype=float) for p in predictors]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
