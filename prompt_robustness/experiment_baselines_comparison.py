"""
experiment_baselines_comparison.py — head-to-head: LL-PIRC vs 4 no-training
mitigation baselines on the same article set.

This is the HEADLINE LL-PIRC paper result (PUBLICATION_SPEC §10, Table 2).
Without this table, the LL-PIRC method paper has no defensible "we beat the
strongest no-training intervention" claim — reviewers will (correctly) ask
"is your gain just because any test-time intervention reduces variance?"

The 4 baselines come from `src/mitigation_baselines.py`:
  1. temperature_smoothing   — sample K outputs at T=0.7 per paraphrase, take longest
  2. self_consistency_vote   — ROUGE-L centroid across paraphrases
  3. system_prompt_stabilize — "be stable across rephrasings" system prompt
  4. in_context_learning     — K-1 paraphrase exemplars in context

PIRC results are reused from a prior `pirc.json` (produced by
`experiment_pirc.py`) when present, so we never re-run PIRC unnecessarily.

OUTPUTS
-------
  results/baselines_comparison.json         per-article + summary per method
  results/baselines_comparison_summary.md   human-readable headline table

USAGE
-----
  # Standard run (assumes baseline.json + pirc.json are already on disk)
  python prompt_robustness/experiment_baselines_comparison.py \\
      --config prompt_robustness/config.yaml \\
      --results-dir results \\
      --baselines temperature_smoothing self_consistency_vote system_prompt_stabilize in_context_learning

  # Restrict to a subset of methods, e.g. for a quick PIRC vs ICL only
  python prompt_robustness/experiment_baselines_comparison.py \\
      --config prompt_robustness/config.yaml \\
      --results-dir results \\
      --baselines in_context_learning

  # Dry-run: just print the article schedule without loading the model
  python prompt_robustness/experiment_baselines_comparison.py \\
      --config prompt_robustness/config.yaml --results-dir results --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional

import torch

# Reuse the helpers from experiment_pirc.py rather than reimplementing them.
# This keeps the article schedule, prompt formatting, and ROUGE scoring
# identical to the PIRC harness.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "src"))

from experiment_pirc import (  # type: ignore  noqa: E402
    load_config,
    load_baseline_results,
    load_target_model,
    build_prompts_from_baseline_result,
    score_outputs_rouge_l,
    compute_rouge_l,
)
from mitigation_baselines import BASELINES  # type: ignore  noqa: E402

logger = logging.getLogger("baselines_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Generation surface — wraps HF model.generate into the (prompt, kwargs) → str
# signature expected by every function in mitigation_baselines.py.
# ─────────────────────────────────────────────────────────────────────────────

def build_generate_fn(
    model,
    tokenizer,
    default_max_new_tokens: int = 256,
) -> Callable[[str, Dict[str, Any]], str]:
    """Return a closure `(prompt, kwargs) -> generated_text` that handles:
      - greedy vs sampled decoding (gen_kwargs.do_sample)
      - per-call seed for sampling reproducibility (gen_kwargs.seed)
      - optional system_prompt prepended via the tokenizer's chat template
        when present (graceful fallback to raw concatenation otherwise)
    """
    device = next(model.parameters()).device

    def _generate(prompt: str, gen_kwargs: Optional[Dict[str, Any]] = None) -> str:
        gen_kwargs = gen_kwargs or {}
        max_new = int(gen_kwargs.get("max_new_tokens", default_max_new_tokens))
        do_sample = bool(gen_kwargs.get("do_sample", False))
        temperature = float(gen_kwargs.get("temperature", 1.0))
        seed = gen_kwargs.get("seed")
        system_prompt = gen_kwargs.get("system_prompt")

        # Optional system prompt path — preferred when the tokenizer ships a
        # chat template; otherwise fall back to a single text concatenation
        # that mirrors what experiment_pirc.py does for raw prompts.
        if system_prompt and hasattr(tokenizer, "apply_chat_template"):
            try:
                msg = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ]
                input_ids = tokenizer.apply_chat_template(
                    msg, return_tensors="pt", add_generation_prompt=True
                ).to(device)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"chat_template path failed ({e}); falling back to concat")
                full = f"{system_prompt}\n\n{prompt}"
                input_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
        else:
            full = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
            input_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)

        prompt_len = input_ids.shape[-1]

        gen_args: Dict[str, Any] = {
            "input_ids":      input_ids,
            "max_new_tokens": max_new,
            "do_sample":      do_sample,
            "pad_token_id":   tokenizer.pad_token_id,
        }
        if do_sample:
            gen_args["temperature"] = temperature
            if seed is not None:
                # Per-call RNG seed for reproducible sampling.
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            out = model.generate(**gen_args)
        new_tokens = out[0][prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


# ─────────────────────────────────────────────────────────────────────────────
# Per-article: run all selected baselines + score against gold
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines_for_article(
    br: Dict[str, Any],
    config: Dict[str, Any],
    generate_fn: Callable,
    selected_baselines: List[str],
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Run each selected baseline on this article's K paraphrase prompts.

    Returns: per-article record with per-baseline outputs + ROUGE stats.
    """
    prompts = build_prompts_from_baseline_result(br, config)
    gold = br.get("gold_summary", "")

    per_method: Dict[str, Any] = {}
    for name in selected_baselines:
        if name not in BASELINES:
            logger.warning(f"  unknown baseline '{name}' — skipping")
            continue
        fn = BASELINES[name]
        t0 = time.time()
        try:
            outputs = fn(prompts, generate_fn, max_new_tokens=max_new_tokens)
        except TypeError:
            # Older signature without explicit max_new_tokens kwarg.
            outputs = fn(prompts, generate_fn)
        elapsed = time.time() - t0

        rouge_l_per = score_outputs_rouge_l(outputs, gold)
        rouge_mean = float(mean(rouge_l_per)) if rouge_l_per else 0.0
        rouge_var  = float(stdev(rouge_l_per) ** 2) if len(rouge_l_per) >= 2 else 0.0
        rouge_min  = float(min(rouge_l_per)) if rouge_l_per else 0.0

        per_method[name] = {
            "outputs":        outputs,
            "rouge_l_scores": rouge_l_per,
            "rouge_mean":     rouge_mean,
            "rouge_var":      rouge_var,
            "rouge_min":      rouge_min,
            "wall_time_s":    round(elapsed, 2),
        }
        logger.info(
            f"  {name:28s}  ROUGE-L mean={rouge_mean:.4f}  var={rouge_var:.2e}  "
            f"min={rouge_min:.4f}  time={elapsed:.1f}s"
        )
    return per_method


# ─────────────────────────────────────────────────────────────────────────────
# PIRC results reuse — pull from prior pirc.json so we never re-run PIRC.
# ─────────────────────────────────────────────────────────────────────────────

def load_pirc_results(results_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Map article_idx → PIRC per-article record from pirc.json."""
    pirc_path = results_dir / "pirc.json"
    if not pirc_path.exists():
        logger.warning(f"  no PIRC results at {pirc_path} — comparison will omit PIRC")
        return {}
    with open(pirc_path) as f:
        data = json.load(f)
    by_idx: Dict[int, Dict[str, Any]] = {}
    for r in data.get("results", []):
        if r.get("article_idx") is not None:
            by_idx[r["article_idx"]] = r
    logger.info(f"  loaded PIRC results for {len(by_idx)} articles from {pirc_path}")
    return by_idx


def pirc_record_to_method_block(pirc_rec: Dict[str, Any], gold: str) -> Dict[str, Any]:
    """Adapt a per-article PIRC record into the same shape as a baseline block
    so the comparison summarizer can treat all methods uniformly."""
    rouge_l_per = pirc_rec.get("pirc_rouge_scores")
    if rouge_l_per is None:
        outs = pirc_rec.get("pirc_outputs") or []
        rouge_l_per = [compute_rouge_l(o or "", gold) for o in outs]
    rouge_l_per = [float(x) for x in rouge_l_per]
    return {
        "outputs":        pirc_rec.get("pirc_outputs") or [],
        "rouge_l_scores": rouge_l_per,
        "rouge_mean":     float(mean(rouge_l_per)) if rouge_l_per else 0.0,
        "rouge_var":      float(stdev(rouge_l_per) ** 2) if len(rouge_l_per) >= 2 else 0.0,
        "rouge_min":      float(min(rouge_l_per)) if rouge_l_per else 0.0,
        "wall_time_s":    None,  # not recorded by experiment_pirc.py
        "alpha_used":     pirc_rec.get("alpha_used"),
        "tau_used":       pirc_rec.get("tau_used"),
        "ell_star":       pirc_rec.get("ell_star"),
        "num_anchors":    pirc_rec.get("num_anchors"),
    }


def baseline_record_to_method_block(br: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt the no-intervention baseline record (from baseline.json)."""
    rouge_l_per = [float(x) for x in (br.get("rouge_scores") or [])]
    return {
        "outputs":        br.get("outputs") or [],
        "rouge_l_scores": rouge_l_per,
        "rouge_mean":     float(mean(rouge_l_per)) if rouge_l_per else 0.0,
        "rouge_var":      float(stdev(rouge_l_per) ** 2) if len(rouge_l_per) >= 2 else 0.0,
        "rouge_min":      float(min(rouge_l_per)) if rouge_l_per else 0.0,
        "wall_time_s":    None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_comparison(per_article: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each method, compute summary stats across all articles + the
    "variance reduction vs baseline (no intervention)" headline number.

    The variance-reduction metric matches `experiment_pirc.py`:
        var_reduction = 1 - mean(var_method) / mean(var_baseline_no_intervention)
    """
    if not per_article:
        return {"n_articles": 0}

    # Collect method-keyed lists across articles.
    methods: Dict[str, Dict[str, List[float]]] = {}
    for art in per_article:
        for name, blk in (art.get("methods") or {}).items():
            d = methods.setdefault(name, {
                "rouge_mean": [], "rouge_var": [], "rouge_min": [], "wall_time": [],
            })
            d["rouge_mean"].append(blk["rouge_mean"])
            d["rouge_var"].append(blk["rouge_var"])
            d["rouge_min"].append(blk["rouge_min"])
            if blk.get("wall_time_s") is not None:
                d["wall_time"].append(blk["wall_time_s"])

    baseline_key = "baseline_no_intervention"
    baseline_mean_var = (
        mean(methods[baseline_key]["rouge_var"]) if baseline_key in methods else None
    )

    summary: Dict[str, Any] = {}
    for name, vals in methods.items():
        rouge_mean_mu = float(mean(vals["rouge_mean"]))
        rouge_var_mu  = float(mean(vals["rouge_var"]))
        rouge_min_mu  = float(mean(vals["rouge_min"]))
        worst_overall = float(min(vals["rouge_min"])) if vals["rouge_min"] else 0.0
        var_reduction = (
            (1.0 - rouge_var_mu / baseline_mean_var)
            if (baseline_mean_var is not None and baseline_mean_var > 0)
            else None
        )
        summary[name] = {
            "n_articles":               len(vals["rouge_mean"]),
            "mean_rouge_l":             rouge_mean_mu,
            "mean_rouge_l_variance":    rouge_var_mu,
            "mean_worst_prompt_rouge":  rouge_min_mu,
            "worst_overall_rouge":      worst_overall,
            "variance_reduction_vs_baseline": var_reduction,
            "mean_wall_time_s":         float(mean(vals["wall_time"])) if vals["wall_time"] else None,
        }
    return {
        "n_articles":  len(per_article),
        "per_method":  summary,
        "baseline_key": baseline_key,
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    """Headline table — the one that goes verbatim in the paper."""
    pm = summary.get("per_method", {})
    if not pm:
        return "(no methods to compare)"

    # Method ordering: baseline first, baselines (alphabetical), PIRC last.
    order = ["baseline_no_intervention"]
    standard = sorted(k for k in pm if k not in {"baseline_no_intervention", "pirc"})
    order += standard
    if "pirc" in pm:
        order.append("pirc")
    order = [k for k in order if k in pm]

    pretty = {
        "baseline_no_intervention": "Baseline (no intervention)",
        "temperature_smoothing":    "Temperature smoothing (T=0.7, n=4)",
        "self_consistency_vote":    "Self-consistency vote",
        "system_prompt_stabilize":  "System-prompt stabilization",
        "in_context_learning":      "In-context paraphrase exemplars",
        "pirc":                     "LL-PIRC (ours)",
    }

    md: List[str] = []
    md.append("# LL-PIRC vs Mitigation Baselines — Headline Comparison\n")
    md.append(f"_n articles = {summary['n_articles']}_\n")
    md.append("| Method | Mean ROUGE-L | Variance ↓ | Mean worst-prompt ROUGE-L | Variance reduction vs baseline |")
    md.append("|---|---|---|---|---|")
    for k in order:
        v = pm[k]
        var_red = v["variance_reduction_vs_baseline"]
        var_red_s = f"{100*var_red:.1f}%" if isinstance(var_red, float) else "—"
        md.append(
            f"| {pretty.get(k, k)} | {v['mean_rouge_l']:.4f} | "
            f"{v['mean_rouge_l_variance']:.2e} | {v['mean_worst_prompt_rouge']:.4f} | {var_red_s} |"
        )
    md.append("")
    md.append("> Variance reduction = `1 - mean(var_method) / mean(var_baseline)`")
    md.append("> Worst-prompt ROUGE = `min_k ROUGE-L(output_k, gold)` per article")
    md.append("> Baseline (no intervention) is the un-mitigated subject-model output.")
    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="prompt_robustness/config.yaml")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--baselines", nargs="+",
                   default=list(BASELINES.keys()),
                   choices=list(BASELINES.keys()),
                   help="Which baselines to run (default: all 4)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-articles", type=int, default=0,
                   help="Cap on number of baseline articles processed (0 = all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the schedule without loading the target model")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_argparser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "baselines_comparison.json"
    out_md   = results_dir / "baselines_comparison_summary.md"

    config = load_config(args.config)
    baseline_doc = load_baseline_results(str(results_dir))
    if baseline_doc is None or not baseline_doc.get("results"):
        sys.exit(f"ERROR: no baseline.json results in {results_dir}. "
                 "Run experiment_baseline.py first.")
    articles = baseline_doc["results"]
    if args.max_articles > 0:
        articles = articles[: args.max_articles]

    print(f"→ {len(articles)} articles scheduled across {len(args.baselines)} baseline(s) + PIRC")
    print(f"  baselines: {args.baselines}")
    pirc_by_idx = load_pirc_results(results_dir)

    if args.dry_run:
        print("\n[dry-run] would now load target model and generate. exiting.")
        return 0

    # Resume support: if a comparison JSON already exists, pick up where it stopped.
    per_article_existing: List[Dict[str, Any]] = []
    done_idxs: set = set()
    if out_json.exists():
        try:
            with open(out_json) as f:
                prev = json.load(f)
            per_article_existing = prev.get("per_article") or []
            done_idxs = {r["article_idx"] for r in per_article_existing
                         if r.get("article_idx") is not None}
            if done_idxs:
                print(f"↻ Resuming: {len(done_idxs)} articles already on disk")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not parse existing {out_json}: {e}")

    print("\n→ Loading target model …")
    model, tokenizer = load_target_model(config)
    generate_fn = build_generate_fn(model, tokenizer,
                                    default_max_new_tokens=args.max_new_tokens)

    per_article: List[Dict[str, Any]] = list(per_article_existing)
    for i, br in enumerate(articles, 1):
        a_idx = br.get("article_idx")
        if a_idx in done_idxs:
            continue
        print(f"\n── Article {i}/{len(articles)} (idx={a_idx}) ──")
        methods_block: Dict[str, Any] = {}

        # 1. The no-intervention baseline is already in baseline.json — adopt it.
        methods_block["baseline_no_intervention"] = baseline_record_to_method_block(br)

        # 2. Run the selected mitigation baselines fresh.
        methods_block.update(
            run_baselines_for_article(
                br, config, generate_fn, args.baselines, args.max_new_tokens,
            )
        )

        # 3. PIRC: reuse the saved per-article record when available.
        if a_idx in pirc_by_idx:
            methods_block["pirc"] = pirc_record_to_method_block(
                pirc_by_idx[a_idx], br.get("gold_summary", "")
            )
        else:
            logger.warning(f"  no PIRC record for article {a_idx} — omitting from this row")

        per_article.append({
            "article_idx":   a_idx,
            "article_id":    br.get("article_id"),
            "n_paraphrases": len(br.get("paraphrase_instructions") or []),
            "methods":       methods_block,
        })

        # Persist incrementally so a mid-run crash is recoverable.
        summary = aggregate_comparison(per_article)
        with open(out_json, "w") as f:
            json.dump({
                "config_path":  args.config,
                "results_dir":  str(results_dir),
                "baselines":    args.baselines,
                "n_articles":   len(per_article),
                "per_article":  per_article,
                "summary":      summary,
            }, f, indent=2)

    # Final rendering.
    summary = aggregate_comparison(per_article)
    with open(out_md, "w") as f:
        f.write(render_markdown(summary))

    print(f"\n✓ wrote {out_json}")
    print(f"✓ wrote {out_md}")
    print("\n" + "=" * 72)
    print(" HEADLINE COMPARISON (mean over articles)")
    print("=" * 72)
    pm = summary.get("per_method", {})
    for k, v in pm.items():
        red = v["variance_reduction_vs_baseline"]
        red_s = f"{100*red:5.1f}%" if isinstance(red, float) else "  n/a"
        print(f"  {k:28s}  ROUGE={v['mean_rouge_l']:.4f}  var↓={red_s}  "
              f"worst={v['mean_worst_prompt_rouge']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
