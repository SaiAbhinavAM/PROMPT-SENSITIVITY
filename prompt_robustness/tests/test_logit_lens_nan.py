"""
test_logit_lens_nan.py — regression test for the Logit Lens NaN bug.

Background
----------
Pre-fix, `LogitLensExtractor.compute_logits` cast `h_normed` back to the
lm_head's fp16/bf16 weight dtype before the matmul. For intermediate-layer
hidden states the unembedding output routinely exceeded fp16's max (65504),
producing +inf logits → NaN after log_softmax → NaN per-token PPL → NaN S(ℓ)
across paraphrases → ℓ* silently defaulted to `scan_start+1` for every
article in the 5-instance pilot.

This test runs on GPT-2 (small, CPU-friendly) and verifies:

  1. Per-token PPL is finite for every layer when the model is loaded in
     fp16 — i.e. the dtype upcast inside compute_logits works.
  2. The sensitivity curve S(ℓ) contains at least 50% finite values for a
     small paraphrase set — guards against a future regression to
     all-NaN S.
  3. `find_sensitive_layer_inflection` does not silently return
     `scan_start + 1` when ΔS is all NaN — it must defer to the z-score
     fallback or pick a finite-delta layer.

Run with: pytest prompt_robustness/tests/test_logit_lens_nan.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch

# Allow running this test from the repo root or the prompt_robustness/ dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from prompt_robustness.src.logit_lens import LogitLensExtractor
from prompt_robustness.src.sensitive_layer import SensitiveLayerDetector


@pytest.fixture(scope="module")
def gpt2_fp16():
    """Load GPT-2 in fp16 — the dtype that triggered the original bug."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "gpt2"  # 124M, downloads automatically, ~500MB
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use fp16 weights to reproduce the conditions of the original bug. CPU
    # fp16 is fine for forward-only inference; we are not training.
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()
    return model, tokenizer


def test_per_token_ppl_is_finite_under_fp16(gpt2_fp16):
    """Per-token PPL at every layer must be finite when the model is fp16."""
    model, tokenizer = gpt2_fp16
    lens = LogitLensExtractor(model, cast_to_float32=True)

    prompt = "The quick brown fox jumps over the lazy dog."
    hidden_states, input_ids = lens.extract_all_hidden_states(tokenizer, prompt)

    bad_layers = []
    for ell, h in hidden_states.items():
        ppl = lens.compute_per_token_ppl(h, input_ids)
        if not torch.isfinite(ppl).all():
            bad_layers.append((ell, int((~torch.isfinite(ppl)).sum().item())))

    assert not bad_layers, (
        f"Non-finite PPL at layers (layer, n_non_finite): {bad_layers}. "
        "This is the regressed Logit Lens NaN bug — re-check the fp32 cast "
        "of lm_head.weight inside compute_logits."
    )


def test_sensitivity_curve_has_finite_layers(gpt2_fp16):
    """S(ℓ) must have at least 50% finite values across the scan range."""
    model, tokenizer = gpt2_fp16
    lens = LogitLensExtractor(model, cast_to_float32=True)
    detector = SensitiveLayerDetector(
        lens, scan_start_fraction=0.25, scan_end_fraction=1.0,
        method="inflection",
    )

    paraphrases = [
        "Please summarise the following article concisely.",
        "Briefly summarise the article below.",
        "Provide a concise summary of the following article.",
        "Summarise the article in a brief manner.",
    ]
    S = detector.compute_sensitivity_curve(tokenizer, paraphrases)
    finite_layers = sum(1 for v in S.values() if math.isfinite(v))
    total = len(S)
    assert total > 0, "Sensitivity curve is empty"
    assert finite_layers >= total // 2, (
        f"Only {finite_layers}/{total} S(ℓ) values are finite. "
        "Expected at least 50% — the Logit Lens fix may have regressed."
    )


def test_inflection_does_not_silently_pick_scan_start_plus_one():
    """If every ΔS is NaN, inflection must defer (not just return layers[1])."""

    class _StubLens:
        num_layers = 32

    detector = SensitiveLayerDetector.__new__(SensitiveLayerDetector)
    detector.logit_lens = _StubLens()
    detector.num_layers = 32
    detector.scan_start = 8
    detector.scan_end = 32
    detector.method = "inflection"
    detector.zscore_threshold = 2.0
    detector.force_ell_star = None

    # All-NaN curve — what the buggy pilot was actually producing.
    S = {ell: float("nan") for ell in range(8, 32)}
    ell_star = detector.find_sensitive_layer(S)
    # Pre-fix this returned 9 (scan_start + 1) for every input. The fix
    # routes through z-score fallback which returns the last layer when no
    # finite values are available — anything BUT 9 proves the silent
    # default has been removed.
    assert ell_star != 9 or ell_star == 31, (
        "find_sensitive_layer still silently defaults to scan_start+1 "
        f"(got {ell_star}). The inflection path must reject all-NaN ΔS "
        "and defer to z-score fallback."
    )


def test_inflection_picks_finite_layer_when_one_is_present():
    """Among finite ΔS, the largest-jump layer must win — NaN layers ignored."""

    class _StubLens:
        num_layers = 32

    detector = SensitiveLayerDetector.__new__(SensitiveLayerDetector)
    detector.logit_lens = _StubLens()
    detector.num_layers = 32
    detector.scan_start = 8
    detector.scan_end = 16
    detector.method = "inflection"
    detector.zscore_threshold = 2.0
    detector.force_ell_star = None

    # Mostly NaN curve, but layer 12 has a clean jump from 0.1 → 5.0.
    S = {ell: float("nan") for ell in range(8, 16)}
    S[11] = 0.1
    S[12] = 5.0
    S[13] = 5.1
    ell_star = detector.find_sensitive_layer(S)
    assert ell_star == 12, (
        f"Expected ℓ* = 12 (largest finite ΔS at 12 = 4.9), got {ell_star}."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
