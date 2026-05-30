"""
mitigation_baselines.py — Alternative robustness interventions for the
LL-PIRC method paper (Flaw §5.5 baselines).

For the method paper, LL-PIRC must beat at least the strongest of these
no-training-required baselines on variance reduction at preserved quality:

  1. temperature_smoothing  — average K samples at T=0.7 per paraphrase
  2. self_consistency_vote  — modal answer across paraphrases (classification
                              tasks) or longest-common-subsequence consensus
                              (generation tasks)
  3. system_prompt_stabilize — prepend a "be stable across rephrasings"
                              system prompt to every paraphrase variant
  4. in_context_learning    — show K paraphrase examples + their consensus
                              answer in-context, then ask the real question

Each function takes the same signature so they can be plugged into the same
evaluation harness as `generate_with_clamping` (PIRC). All four are
intentionally inference-time only — no fine-tuning, no extra labels.

Wire them in via a new `experiment_baselines_comparison.py` (TBD) that runs
all 5 interventions (4 baselines + PIRC) on the same article set and reports
ROUGE-L variance reduction + mean quality.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1 — Temperature smoothing
# ─────────────────────────────────────────────────────────────────────────────

def temperature_smoothing(
    prompts: Sequence[str],
    generate_fn: Callable[[str, dict], str],
    *,
    n_samples: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    aggregator: Callable[[List[str]], str] = None,
) -> List[str]:
    """For each of K paraphrases, sample n_samples outputs at T=0.7 and
    return the per-paraphrase aggregate (default: longest sample).

    Args:
        prompts: K paraphrase prompts.
        generate_fn: (prompt, gen_kwargs) -> str. Caller binds the model.
        n_samples: K_samples per paraphrase (default 4).
        temperature: sampling temperature.
        max_new_tokens: per-sample cap.
        aggregator: how to collapse n_samples → 1 output (default: longest).

    Returns:
        K outputs (one per paraphrase).
    """
    if aggregator is None:
        aggregator = lambda samples: max(samples, key=len) if samples else ""

    outputs: List[str] = []
    for k, prompt in enumerate(prompts):
        samples: List[str] = []
        for s in range(n_samples):
            text = generate_fn(prompt, {
                "do_sample": True,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                # Different seed per sample so we get diverse rollouts.
                "seed": 42 + 1000 * k + s,
            })
            samples.append(text)
        outputs.append(aggregator(samples))
        logger.debug(f"  smoothing paraphrase {k+1}/{len(prompts)}: "
                     f"{n_samples} samples → len={len(outputs[-1])}")
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2 — Self-consistency voting across paraphrases
# ─────────────────────────────────────────────────────────────────────────────

def self_consistency_vote(
    prompts: Sequence[str],
    generate_fn: Callable[[str, dict], str],
    *,
    max_new_tokens: int = 256,
    classification: bool = False,
) -> List[str]:
    """Generate one greedy output per paraphrase, then vote.

    Classification mode (`classification=True`): the modal answer (most
    common output string after light normalization) is returned for every
    position — variance ≈ 0 by construction except for ties.

    Generation mode (default): pick the response whose ROUGE-L mean to the
    other K-1 is highest — the "central" generation. All K positions then
    return this same consensus generation, so variance collapses but the
    K=1 effective sample size means ROUGE-L vs gold is whatever that one
    centroid response scores.
    """
    raw = [generate_fn(p, {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }) for p in prompts]

    if classification:
        normed = [r.strip().lower() for r in raw]
        if not normed:
            return []
        modal = Counter(normed).most_common(1)[0][0]
        # Map the modal back to one original surface form (first match).
        modal_surface = next((r for r in raw if r.strip().lower() == modal), raw[0])
        return [modal_surface] * len(raw)

    # Generation mode: pick centroid by ROUGE-L mean. Lazy import so this
    # module loads without rouge_score installed.
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for i, ri in enumerate(raw):
        other = [raw[j] for j in range(len(raw)) if j != i]
        if not other or not ri.strip():
            scores.append(0.0)
            continue
        mean_r = sum(
            sc.score(ri, rj)["rougeL"].fmeasure for rj in other
        ) / max(1, len(other))
        scores.append(mean_r)
    centroid = raw[max(range(len(raw)), key=lambda i: scores[i])] if raw else ""
    return [centroid] * len(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3 — System-prompt stabilization
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STABILITY_SYSTEM_PROMPT = (
    "You are a careful assistant. Different users may phrase the same "
    "request in different ways; your answer should be identical regardless "
    "of how the question is worded. Focus on the underlying meaning."
)


def system_prompt_stabilize(
    prompts: Sequence[str],
    generate_fn: Callable[[str, dict], str],
    *,
    system_prompt: str = DEFAULT_STABILITY_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
) -> List[str]:
    """Prepend a stability-focused system prompt to every paraphrase and
    generate greedily. The system prompt is passed via `gen_kwargs` so the
    model interface can apply its chat template properly."""
    outputs: List[str] = []
    for k, prompt in enumerate(prompts):
        text = generate_fn(prompt, {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "system_prompt": system_prompt,
        })
        outputs.append(text)
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4 — In-context learning with K-1 paraphrase exemplars
# ─────────────────────────────────────────────────────────────────────────────

def in_context_learning(
    prompts: Sequence[str],
    generate_fn: Callable[[str, dict], str],
    *,
    max_new_tokens: int = 256,
    exemplar_answer: Optional[str] = None,
) -> List[str]:
    """For each prompt position i, build an ICL context made of the OTHER
    K-1 paraphrases. If `exemplar_answer` is provided (e.g. gold output for
    a held-out instance), show it as the demonstration answer to anchor the
    style; otherwise leave the demonstrations open-ended so the model only
    sees that "all these phrasings should yield the same answer".
    """
    outputs: List[str] = []
    for i, prompt in enumerate(prompts):
        exemplars = []
        for j, other in enumerate(prompts):
            if j == i:
                continue
            block = f"Example phrasing:\n{other}"
            if exemplar_answer:
                block += f"\nAnswer: {exemplar_answer}"
            exemplars.append(block)
        icl_prefix = "\n\n".join(exemplars)
        composite = (
            f"{icl_prefix}\n\nThe phrasing below is asking the same thing.\n"
            f"Give the SAME answer.\n\n{prompt}"
        )
        outputs.append(generate_fn(composite, {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        }))
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Registry for experiment_baselines_comparison.py
# ─────────────────────────────────────────────────────────────────────────────

BASELINES = {
    "temperature_smoothing": temperature_smoothing,
    "self_consistency_vote": self_consistency_vote,
    "system_prompt_stabilize": system_prompt_stabilize,
    "in_context_learning": in_context_learning,
}
