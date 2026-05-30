"""
adversarial_paraphrases.py — Flaw §6.2 adversarial paraphrase subset.

GenSens currently produces only "easy" LLM-rewritten paraphrases. For a
clean "robustness under attack" story the benchmark also needs a hard
adversarial subset. This module provides five deterministic perturbation
families that operate without a model:

    1. typo                — single-character swaps / drops in 5-10% of words
    2. sentence_reorder    — shuffle independent clauses (split on ', ' /
                             '. ' / ';')
    3. double_negation     — "X is Y" → "X is not not Y" (semantics
                             preserved, lexical surface heavily perturbed)
    4. hedged              — "X is Y" → "X might be Y / could possibly be Y"
    5. formality_shift     — flip "do not" ↔ "don't", colloquial ↔ formal
                             contraction substitutions

Each function takes (text: str, seed: int) → str so the perturbations are
reproducible. The `adversarial_paraphrases(text, n, seed)` driver returns
up to `n` distinct adversarial variants by combining strategies.

Integrate by treating adversarial variants as an extra strategy bucket in
GenSens output; report PRI separately on the easy vs adversarial subset.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy perturbations
# ─────────────────────────────────────────────────────────────────────────────

def typo(text: str, seed: int = 0, rate: float = 0.08) -> str:
    """Apply single-character swaps in approximately `rate` of word tokens."""
    rng = random.Random(seed)
    words = text.split()
    n_perturb = max(1, int(round(len(words) * rate)))
    indices = rng.sample(range(len(words)), min(n_perturb, len(words)))
    for i in indices:
        w = words[i]
        if len(w) < 4:
            continue
        # Swap two adjacent middle characters.
        j = rng.randrange(1, len(w) - 2)
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        words[i] = w
    return " ".join(words)


def sentence_reorder(text: str, seed: int = 0) -> str:
    """Split on sentence-or-clause boundaries and shuffle the pieces."""
    rng = random.Random(seed)
    parts = re.split(r"(?<=[.!?])\s+|;\s+", text.strip())
    parts = [p for p in parts if p]
    if len(parts) < 2:
        # Fall back to comma-clause split.
        parts = [p.strip() for p in text.split(", ") if p.strip()]
    if len(parts) < 2:
        return text
    rng.shuffle(parts)
    out = ". ".join(parts)
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


_DBL_NEG_RE = re.compile(r"\b(is|are|was|were|am|be|been|being)\b", re.IGNORECASE)


def double_negation(text: str, seed: int = 0) -> str:
    """Insert 'not not' after the first copula occurrence. Semantically a
    no-op (¬¬X ≡ X) but surface form is heavily perturbed."""
    m = _DBL_NEG_RE.search(text)
    if not m:
        return text
    end = m.end()
    return text[:end] + " not not" + text[end:]


_HEDGE_PHRASES = [
    "It seems that",
    "It might be the case that",
    "Arguably,",
    "Some might say",
    "It could be argued that",
]


def hedged(text: str, seed: int = 0) -> str:
    """Prepend a hedging phrase. Semantically conservative; surface diverges."""
    rng = random.Random(seed)
    hedge = rng.choice(_HEDGE_PHRASES)
    # Lowercase first letter of original to read naturally after the hedge.
    if text:
        text = text[0].lower() + text[1:]
    return f"{hedge} {text}"


_CONTRACTIONS: Dict[str, str] = {
    " do not ": " don't ",
    " does not ": " doesn't ",
    " did not ": " didn't ",
    " is not ": " isn't ",
    " are not ": " aren't ",
    " was not ": " wasn't ",
    " were not ": " weren't ",
    " cannot ": " can't ",
    " will not ": " won't ",
    " would not ": " wouldn't ",
    " I am ": " I'm ",
    " you are ": " you're ",
    " they are ": " they're ",
    " we are ": " we're ",
    " it is ": " it's ",
}


def formality_shift(text: str, seed: int = 0) -> str:
    """Flip contractions. If text contains expanded forms, contract them;
    if it contains contracted forms, expand them. Pick whichever direction
    has more matches in the input so output is meaningfully different."""
    rng = random.Random(seed)
    padded = f" {text} "
    expand_hits = sum(1 for k in _CONTRACTIONS if k in padded)
    contract_hits = sum(1 for v in _CONTRACTIONS.values() if v in padded)
    if expand_hits >= contract_hits:
        for k, v in _CONTRACTIONS.items():
            padded = padded.replace(k, v)
    else:
        for k, v in _CONTRACTIONS.items():
            padded = padded.replace(v, k)
    return padded.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Registry + driver
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, Callable[[str, int], str]] = {
    "typo": typo,
    "sentence_reorder": sentence_reorder,
    "double_negation": double_negation,
    "hedged": hedged,
    "formality_shift": formality_shift,
}


def adversarial_paraphrases(text: str, n: int = 5, seed: int = 0) -> List[Dict[str, str]]:
    """Return up to `n` distinct adversarial variants of `text`.

    Cycles through STRATEGIES in order; if `n` > len(STRATEGIES) wraps
    with a different seed each time. Output records carry the strategy
    name so downstream stratified scoring can split easy vs hard.
    """
    out: List[Dict[str, str]] = []
    names = list(STRATEGIES.keys())
    rng = random.Random(seed)
    for i in range(n):
        strat = names[i % len(names)]
        sub_seed = seed * 7919 + i
        variant = STRATEGIES[strat](text, sub_seed)
        # Skip no-ops to avoid contaminating the adversarial bucket.
        if variant.strip() and variant.strip() != text.strip():
            out.append({
                "strategy": f"adv:{strat}",
                "paraphrased_text": variant,
                "seed": sub_seed,
            })
    return out


if __name__ == "__main__":
    # Tiny CLI for ad-hoc inspection.
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else (
        "Summarize the article concisely. The model should not skip any key facts."
    )
    for v in adversarial_paraphrases(sample, n=5, seed=42):
        print(f"[{v['strategy']:25s}] {v['paraphrased_text']}")
