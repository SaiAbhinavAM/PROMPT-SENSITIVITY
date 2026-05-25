#!/usr/bin/env python3
"""
LL-PIRC Local Smoke Test
========================
Validates the full LL-PIRC pipeline end-to-end using gpt2-medium:
  Logit Lens → Sensitive Layer Detection → Anchor Token ID → PIRC Clamping

The previous smoke test failed silently because the absolute PPL-threshold
anchor selection returned 0 anchors → clamp_hook was a no-op → PIRC output
was identical to a vanilla generation.

This rewrite asserts TWO things that the old test did not:
  (1) num_anchors > 0  (the new percentile-based selector guarantees this)
  (2) clamped_output != unclamped_output  (the intervention actually fires)

Model: gpt2-medium  (~360 MB, GPT-2 style, runs on CPU/MPS)

Usage:
    python local_pirc_smoke_test.py [--soft] [--percentile P]
        --soft        : use soft clamping (alpha=0.5) instead of hard (1.0)
        --percentile  : anchor percentile P (default 30)
"""

import sys
import argparse
import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")

# ─── 5 hand-crafted summarization paraphrase variants ─────────────────────────
PARAPHRASES = [
    "Summarize the following news article in 3–4 sentences, capturing the main events and key details.",
    "The news article below should be summarized in 3–4 sentences. The primary events and important details must be captured.",
    "Provide a concise 3–4 sentence overview of the news article, highlighting the core occurrences and critical information.",
    "Can you give a 3–4 sentence summary of the following article that covers the main events and key points?",
    "As a journalist, write a 3–4 sentence summary of the news article below that captures the major events and essential details.",
]

ARTICLE_SNIPPET = (
    "Scientists at CERN have detected what may be the first evidence of a new subatomic particle. "
    "The potential discovery, made at the Large Hadron Collider, could challenge the Standard Model "
    "of physics. Researchers caution that further experiments are needed to confirm the finding. "
    "If validated, the particle would represent the most significant physics breakthrough in a decade."
)

FULL_PROMPTS = [f"{p}\n\nArticle: {ARTICLE_SNIPPET}\n\nSummary:" for p in PARAPHRASES]

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME           = "gpt2-medium"
MAX_NEW_TOKENS       = 60
ALPHA_HARD           = 1.0
ALPHA_SOFT           = 0.5
ANCHOR_PERCENTILE    = 30.0
SCAN_START_FRAC      = 0.25
K                    = len(FULL_PROMPTS)

PASS_ICON = "✅"
FAIL_ICON = "❌"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_smoke_test(alpha: float, percentile: float):
    """Run the complete LL-PIRC pipeline and report pass/fail per component."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Make src/ importable
    src_dir = Path(__file__).parent / "prompt_robustness"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from src.logit_lens     import LogitLensExtractor
    from src.sensitive_layer import SensitiveLayerDetector
    from src.anchor_tokens  import AnchorTokenIdentifier
    from src.pirc           import PIRCGenerator

    results = {}

    # ── 1. Load model ─────────────────────────────────────────────────────
    print_section("STEP 1 — Load gpt2-medium")
    logger.info(f"Loading {MODEL_NAME} ...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    num_layers = len(model.transformer.h)
    logger.info(f"Model loaded. Layers: {num_layers}")
    results["model_loaded"] = True
    results["num_layers"]   = num_layers
    print(f"{PASS_ICON} Model loaded: {MODEL_NAME} ({num_layers} layers) on {device}")

    # ── 2. Logit Lens ─────────────────────────────────────────────────────
    print_section("STEP 2 — Logit Lens Extractor")
    logit_lens = LogitLensExtractor(model, cast_to_float32=True)
    _, mean_ppl = logit_lens.compute_all_layer_ppl(
        tokenizer, FULL_PROMPTS[0], device
    )
    nonzero_layers = sum(1 for v in mean_ppl.values() if v > 0)
    results["logit_lens_ok"]      = nonzero_layers > 0
    results["nonzero_ppl_layers"] = nonzero_layers
    print(f"{PASS_ICON if results['logit_lens_ok'] else FAIL_ICON} "
          f"Logit Lens: {nonzero_layers}/{num_layers} layers returned non-zero PPL")

    # ── 3. Sensitive Layer Detection ──────────────────────────────────────
    print_section("STEP 3 — Sensitive Layer Detection")
    layer_detector = SensitiveLayerDetector(
        logit_lens,
        scan_start_fraction=SCAN_START_FRAC,
        scan_end_fraction=1.0,
        method="inflection",
        zscore_threshold=2.0,
    )
    ell_star, S_curve = layer_detector.detect(tokenizer, FULL_PROMPTS, device)
    nonzero_s = sum(1 for v in S_curve.values() if v > 0)
    results["ell_star"]   = int(ell_star)
    results["s_curve_ok"] = nonzero_s > 0
    print(f"{PASS_ICON if results['s_curve_ok'] else FAIL_ICON} "
          f"S(ℓ) curve: {nonzero_s}/{len(S_curve)} non-zero entries")
    print(f"{PASS_ICON} ℓ* = {ell_star} ({ell_star/num_layers:.0%} of depth)")

    # ── 4. Anchor Token Identification (PERCENTILE METHOD) ───────────────
    print_section(f"STEP 4 — Anchor Identification (percentile={percentile}%)")
    anchor_identifier = AnchorTokenIdentifier(
        logit_lens,
        anchor_percentile=percentile,
        filter_content_words=True,
    )
    anchor_mask, anchor_stats = anchor_identifier.identify_anchors(
        tokenizer, FULL_PROMPTS, ell_star, device
    )
    num_anchors           = anchor_stats["num_anchors"]
    anchor_fraction       = anchor_stats["anchor_fraction"]
    anchor_tokens_decoded = anchor_stats["anchor_tokens_decoded"]
    anchor_positions      = anchor_stats["anchor_positions"]

    results["num_anchors"]           = num_anchors
    results["anchor_fraction"]       = anchor_fraction
    results["anchor_tokens_decoded"] = anchor_tokens_decoded
    results["anchor_method"]         = anchor_stats["method"]
    results["filtering_applied"]     = anchor_stats["filtering_applied"]
    results["anchors_ok"]            = num_anchors >= 1

    if results["anchors_ok"]:
        print(f"{PASS_ICON} Anchors: {num_anchors} positions found "
              f"({anchor_fraction:.1%} of seq, filter_applied={anchor_stats['filtering_applied']})")
        print(f"\n  Selected anchor tokens (pos → token):")
        for pos, tok in zip(anchor_positions[:20], anchor_tokens_decoded[:20]):
            print(f"    {pos:>3}  →  {tok!r}")
        if len(anchor_positions) > 20:
            print(f"    … ({len(anchor_positions)-20} more)")
    else:
        print(f"{FAIL_ICON} Anchors: 0 found — percentile selector failed "
              f"(should never happen with the new method)")
        # Continue anyway so we still emit diagnostics

    # ── 5. PIRC Clamped Generation + Baseline ─────────────────────────────
    print_section("STEP 5 — PIRC Clamped vs Unclamped Generation")
    pirc_generator = PIRCGenerator(
        model=model,
        tokenizer=tokenizer,
        logit_lens=logit_lens,
        layer_detector=layer_detector,
        anchor_identifier=anchor_identifier,
        alpha=alpha,
        soft_clamp_alpha=ALPHA_SOFT,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # 5a — Run the full clamped pipeline
    logger.info(f"Running full PIRC pipeline (α={alpha}, clamping ENABLED)...")
    pirc_result     = pirc_generator.run_pipeline(
        paraphrases=FULL_PROMPTS,
        device=device,
    )
    clamped_output  = pirc_result["output"]

    # 5b — Run the SAME generation but WITHOUT the clamp hook
    logger.info("Running baseline generation (clamping DISABLED) — same primary prompt, greedy decoding...")
    unclamped_output = pirc_generator.generate_baseline(
        primary_prompt=FULL_PROMPTS[0],
        device=device,
    )

    outputs_differ = (clamped_output.strip() != unclamped_output.strip())

    results["clamped_output"]   = clamped_output
    results["unclamped_output"] = unclamped_output
    results["outputs_differ"]   = outputs_differ
    results["pirc_ok"]          = len(clamped_output.strip()) > 0
    results["pirc_len"]         = len(clamped_output)

    print()
    print(f"{'─'*60}")
    print("  CLAMPED OUTPUT (α={:.2f}):".format(alpha))
    print(f"{'─'*60}")
    print(f"  {clamped_output.strip()!r}")
    print()
    print(f"{'─'*60}")
    print("  UNCLAMPED BASELINE OUTPUT:")
    print(f"{'─'*60}")
    print(f"  {unclamped_output.strip()!r}")
    print()

    if outputs_differ:
        print(f"{PASS_ICON} Outputs DIFFER — PIRC intervention is changing the model's behaviour.")
    else:
        print(f"{FAIL_ICON} PIRC produced IDENTICAL output to baseline — intervention is a no-op.")

    # ── 6. Summary ────────────────────────────────────────────────────────
    print_section("SMOKE TEST SUMMARY")
    checks = [
        ("Model loaded",                results.get("model_loaded",   False)),
        ("Logit Lens non-zero PPL",      results.get("logit_lens_ok",  False)),
        ("S(ℓ) curve non-zero",          results.get("s_curve_ok",     False)),
        ("ℓ* detected",                  results.get("ell_star", -1) >= 0),
        (f"≥ 1 anchor token found",      results.get("anchors_ok",     False)),
        ("PIRC output non-empty",        results.get("pirc_ok",        False)),
        ("Clamped ≠ Unclamped output",   results.get("outputs_differ", False)),
    ]

    all_passed = True
    for name, passed in checks:
        icon = PASS_ICON if passed else FAIL_ICON
        print(f"  {icon}  {name}")
        if not passed:
            all_passed = False

    # The user-required gate: anchors_ok AND outputs_differ
    intervention_real = (
        results.get("anchors_ok", False) and results.get("outputs_differ", False)
    )
    results["intervention_real"] = intervention_real

    print()
    if all_passed:
        print(f"{PASS_ICON} ALL CHECKS PASSED — LL-PIRC pipeline is functional on Mac.")
        print("   The intervention is verifiably changing model output.")
    else:
        print(f"{FAIL_ICON} SOME CHECKS FAILED — see details above before running on H100.\n")

    # ── 7. Save results JSON ──────────────────────────────────────────────
    report_path = Path(__file__).parent / "pirc_smoke_test_results.json"
    save_results = {
        "model_loaded":          results.get("model_loaded", False),
        "num_layers":            results.get("num_layers", 0),
        "logit_lens_ok":         results.get("logit_lens_ok", False),
        "nonzero_ppl_layers":    results.get("nonzero_ppl_layers", 0),
        "ell_star":              results.get("ell_star", -1),
        "s_curve_ok":            results.get("s_curve_ok", False),
        "num_anchors":           results.get("num_anchors", 0),
        "anchor_fraction":       results.get("anchor_fraction", 0.0),
        "anchor_tokens_decoded": results.get("anchor_tokens_decoded", []),
        "anchor_method":         results.get("anchor_method", "unknown"),
        "filtering_applied":     results.get("filtering_applied", False),
        "anchors_ok":            results.get("anchors_ok", False),
        "pirc_ok":               results.get("pirc_ok", False),
        "pirc_len":              results.get("pirc_len", 0),
        "clamped_output":        results.get("clamped_output", ""),
        "unclamped_output":      results.get("unclamped_output", ""),
        "outputs_differ":        results.get("outputs_differ", False),
        "intervention_real":     results.get("intervention_real", False),
        "all_passed":            all_passed,
        "alpha":                 alpha,
        "anchor_percentile":     percentile,
    }
    with open(report_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Results saved to: {report_path}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="LL-PIRC Local Smoke Test (gpt2-medium)")
    parser.add_argument(
        "--soft", action="store_true",
        help="Use soft clamping (alpha=0.5) instead of hard clamping (alpha=1.0)"
    )
    parser.add_argument(
        "--percentile", type=float, default=ANCHOR_PERCENTILE,
        help=f"Anchor percentile P (default: {ANCHOR_PERCENTILE})",
    )
    args = parser.parse_args()

    alpha = ALPHA_SOFT if args.soft else ALPHA_HARD
    print(f"\n{'#'*60}")
    print(f"  LL-PIRC Smoke Test  |  model=gpt2-medium")
    print(f"  α={alpha}  K={K}  anchor_percentile={args.percentile}%")
    print(f"{'#'*60}")

    try:
        passed = run_smoke_test(alpha=alpha, percentile=args.percentile)
        sys.exit(0 if passed else 1)
    except Exception as e:
        logger.exception(f"Smoke test crashed: {e}")
        print(f"\n{FAIL_ICON} FATAL ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
