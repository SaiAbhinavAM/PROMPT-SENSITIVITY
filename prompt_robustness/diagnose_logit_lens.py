"""
diagnose_logit_lens.py — confirm the Logit Lens NaN fix on a real model.

Run this once on the GPU box (or locally on GPT-2) after pulling the fix:

    python prompt_robustness/diagnose_logit_lens.py                       # GPT-2 default
    python prompt_robustness/diagnose_logit_lens.py --model meta-llama/Llama-3.1-8B-Instruct

It loads the target model, computes per-layer PPL across 4 paraphrases of a
short prompt, and prints a table of S(ℓ) values plus the chosen ℓ*. If the
fix is working you will see mostly finite S values and a NON-trivial ℓ*
(NOT scan_start + 1 by default).

Pre-fix expected output (REGRESSION):
    Layer 8: S=nan, Layer 9: S=nan, ..., ℓ* = 9 (scan_start + 1)

Post-fix expected output:
    Layer 8: S=0.42, Layer 9: S=0.55, Layer 10: S=1.21, ..., ℓ* = 12 (or similar)
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch

# Make `prompt_robustness.src.*` importable regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from prompt_robustness.src.logit_lens import LogitLensExtractor
from prompt_robustness.src.sensitive_layer import SensitiveLayerDetector


# Compact paraphrase set — fits on a small GPU and stresses the pipeline.
PARAPHRASES = [
    "Please summarise the following article concisely in three sentences.",
    "Provide a brief, three-sentence summary of the article below.",
    "Summarise the article that follows in a concise three-sentence format.",
    "Give a concise three-sentence summary of the following article.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: float16 — the dtype that triggered the bug)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / mps (default: auto-detect)")
    parser.add_argument("--scan-start-fraction", type=float, default=0.25)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n→ Loading {args.model} ({args.dtype})…")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_kwargs = {"torch_dtype": dtype}
    if token:
        model_kwargs["token"] = token
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device:
        model = model.to(args.device)
    elif torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device
    print(f"  loaded on {device}, n_layers={model.config.num_hidden_layers}")

    lens = LogitLensExtractor(model, cast_to_float32=True)
    detector = SensitiveLayerDetector(
        lens,
        scan_start_fraction=args.scan_start_fraction,
        method="inflection",
    )

    # ── Per-layer PPL sanity check on a single prompt ────────────────────
    print("\n→ Per-layer mean-PPL on prompt[0]:")
    hidden, ids = lens.extract_all_hidden_states(tokenizer, PARAPHRASES[0])
    bad_layers = []
    for ell in sorted(hidden.keys()):
        ppl = lens.compute_per_token_ppl(hidden[ell], ids)
        finite = torch.isfinite(ppl).all().item()
        if not finite:
            bad_layers.append(ell)
        if ell % 4 == 0 or ell == max(hidden.keys()):
            print(f"  layer {ell:3d}: mean_ppl={ppl.mean().item():>12.3e}  "
                  f"finite={finite}")
    if bad_layers:
        print(f"\n✗ FAIL — non-finite PPL at layers: {bad_layers}")
        print("  The Logit Lens dtype upcast did not take effect — "
              "re-check src/logit_lens.py compute_logits.")
        return 1
    print("  ✓ all per-layer PPL values are finite")

    # ── Sensitivity curve + ℓ* detection across paraphrases ──────────────
    print(f"\n→ Computing S(ℓ) across {len(PARAPHRASES)} paraphrases …")
    ell_star, S = detector.detect(tokenizer, PARAPHRASES, device=device)
    finite_layers = sum(1 for v in S.values() if math.isfinite(v))
    print(f"  finite layers in S(ℓ): {finite_layers}/{len(S)}")

    print("\n  S(ℓ) curve:")
    layers = sorted(S.keys())
    for ell in layers:
        v = S[ell]
        marker = "  ←  ℓ*" if ell == ell_star else ""
        if math.isfinite(v):
            print(f"    layer {ell:3d}: S={v:>10.4f}{marker}")
        else:
            print(f"    layer {ell:3d}: S=NaN     {marker}")

    print(f"\n→ ℓ* = {ell_star}")

    # Trivial-default detection: pre-fix this always returned scan_start+1.
    scan_start = int(model.config.num_hidden_layers * args.scan_start_fraction)
    if ell_star == scan_start + 1 and finite_layers < len(S) / 2:
        print("\n⚠  WARNING — ℓ* equals scan_start+1 AND the S-curve is mostly NaN.")
        print("   This is the regressed pre-fix behaviour. Inspect the per-layer")
        print("   PPL block above to see which layers produced NaN.")
        return 2

    print("\n✓ Logit Lens looks healthy — ℓ* was selected from a finite S-curve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
