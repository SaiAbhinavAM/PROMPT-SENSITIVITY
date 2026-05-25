"""
generate_dataset_lightweight.py
================================
CPU-only dataset generator for GenSens — no torch/transformers required.

Uses statistical heuristics to simulate three seq2seq models:
  • flan-t5-small   (~77M params)   ← lightest
  • flan-t5-base    (~250M params)  ← medium
  • distilbart-sim  (~306M params)  ← heaviest (distilbart-like behaviour)

All well under the 2B-parameter ceiling.

Output schema — WIDE format (one row per sample × model):
  topic_label, sample_idx, model,
  input_text, original_prompt,
  prompt_variant_1 … prompt_variant_5,
  model_output_v1 … model_output_v5,
  ppl_v1 … ppl_v5, bf_v1 … bf_v5,
  rouge1_v1…v5, rouge2_v1…v5, rougeL_v1…v5,
  bleu_v1…v5, lex_sim_v1…v5, exact_match_v1…v5,
  avg_ppl, avg_bf, ppl_std, avg_rouge1, avg_rouge2, avg_rougeL,
  avg_bleu, avg_lex_sim, rougeL_std, bleu_std,
  ── PRI Framework ──
  sms, auc_e, trd, kpig,        (ORI sub-metrics)
  ppl_var, pc_stab,             (IFI sub-metrics)
  ori, ifi, pri,                (composite indices)
  cs, hs, diagnosis,            (scores + 2×2 label)
  reference_output

PRI = HM(ORI, IFI)
ORI = HM(norm_sms, norm_auc_e, norm_trd, norm_kpig)
IFI = HM(norm_ppl_var, norm_pc_stab)

Diagnosis 2×2 matrix (ORI × IFI threshold 0.5):
  High ORI + High IFI → True Robustness
  High ORI + Low  IFI → Evaluation Artifact
  Low  ORI + High IFI → Stochastic Luck
  Low  ORI + Low  IFI → Knowledge Boundary

Run:
    python3 generate_dataset_lightweight.py

    OR from the prompt_robustness/ folder:
    ./venv/bin/python3.9 generate_dataset_lightweight.py
"""

import difflib, json, math, random, re, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# ── Evaluation imports ────────────────────────────────────────────────────────
from rouge_score import rouge_scorer as _rouge_mod
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

_ROUGE = _rouge_mod.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
_SMOOTH = SmoothingFunction().method1

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_TASK = 5
SEED             = 42
DATA_DIR         = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# ── Model profiles  (simulate different capacity / style) ─────────────────────
MODEL_PROFILES = {
    "google/flan-t5-small": {
        "params_M":       77,
        "response_len":   (15, 40),     # (min_words, max_words)
        "ppl_base":       18.0,         # base perplexity
        "ppl_spread":     6.0,
        "bf_base":        0.55,         # branching factor base
        "bf_spread":      0.12,
        "vocab_richness": 0.55,         # higher → more varied word choice
        "coherence":      0.60,         # higher → closer to reference
    },
    "google/flan-t5-base": {
        "params_M":       250,
        "response_len":   (25, 60),
        "ppl_base":       12.0,
        "ppl_spread":     4.5,
        "bf_base":        0.65,
        "bf_spread":      0.10,
        "vocab_richness": 0.70,
        "coherence":      0.72,
    },
    "sshleifer/distilbart-cnn-12-6": {
        "params_M":       306,
        "response_len":   (35, 80),
        "ppl_base":       8.5,
        "ppl_spread":     3.0,
        "bf_base":        0.78,
        "bf_spread":      0.08,
        "vocab_richness": 0.82,
        "coherence":      0.83,
    },
}

# ── Prompt variant templates (5 variants per task) ────────────────────────────
# Keys: v1…v5.  v1 is always the "original/base" prompt.
TEMPLATES = {
    "summarization": {
        "v1": "Summarize the following text: {input}",
        "v2": "Condense the main ideas from this passage: {input}",
        "v3": "Produce a concise, neutral summary of the article below: {input}",
        "v4": "Give a brief overview of the key points in this text: {input}",
        "v5": "In one paragraph, capture the essential information from: {input}",
    },
    "code": {
        "v1": "Write a Python function for the following problem: {input}",
        "v2": "Implement a Python solution for this programming task: {input}",
        "v3": "Provide a working Python function that solves the problem described below: {input}",
        "v4": "Create a Python function to address the following specification: {input}",
        "v5": "Code a solution in Python for this task: {input}",
    },
    "creative_writing": {
        "v1": "Write a short story based on this prompt: {input}",
        "v2": "Create a brief narrative inspired by: {input}",
        "v3": "Compose a short creative passage that explores the following idea: {input}",
        "v4": "Craft a short fictional piece using this concept as your starting point: {input}",
        "v5": "Write a vivid short narrative based on: {input}",
    },
    "dialogue": {
        "v1": "Continue this conversation as a helpful assistant: {input}",
        "v2": "Respond naturally to the following dialogue: {input}",
        "v3": "Provide a helpful and coherent reply to the conversation below: {input}",
        "v4": "As a conversational AI, continue the exchange below: {input}",
        "v5": "Give a relevant and informative response to this conversation: {input}",
    },
}

# Original prompt skeleton (no input filled in) — stored in the CSV for reference
ORIGINAL_PROMPT = {
    "summarization":   "Summarize the following text: {input}",
    "code":            "Write a Python function for the following problem: {input}",
    "creative_writing": "Write a short story based on this prompt: {input}",
    "dialogue":        "Continue this conversation as a helpful assistant: {input}",
}

VARIANTS = ["v1", "v2", "v3", "v4", "v5"]

def build_variants(input_text: str, task: str) -> dict:
    truncated = input_text[:600]
    return {v: tmpl.format(input=truncated)
            for v, tmpl in TEMPLATES[task].items()}

# ── Source data ───────────────────────────────────────────────────────────────

SUMMARIZATION_SAMPLES = [
    {
        "input_text": (
            "LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a "
            "reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists "
            "the money won't cast a spell on him. The young actor says he has no plans to fritter "
            "his cash away on fast cars, drink and celebrity parties. 'I don't plan to be one of "
            "those people who, as soon as they turn 18, suddenly buy themselves a massive sports "
            "car collection,' he told an Australian interviewer. Radcliffe's earnings from the "
            "first five Potter films have been held in a trust fund which he has not been able to touch."
        ),
        "reference_output": "Daniel Radcliffe turns 18 and gains access to his £20 million fortune, but says he plans to remain grounded and won't spend it extravagantly.",
    },
    {
        "input_text": (
            "Scientists at the University of Cambridge have developed a new battery technology that "
            "could charge electric vehicles in under five minutes. The breakthrough uses sodium-ion "
            "cells paired with a novel electrode coating that dramatically reduces charge time while "
            "maintaining energy density. The team estimates commercial production could begin within "
            "three years, potentially transforming the EV market."
        ),
        "reference_output": "Cambridge scientists developed a sodium-ion battery technology that can charge EVs in under 5 minutes, with commercial production expected within 3 years.",
    },
    {
        "input_text": (
            "The global coffee market reached a record $100 billion valuation in 2025, driven by "
            "surging demand in Asia and premiumisation trends across Western markets. Vietnam "
            "surpassed Brazil as the largest producer by volume for the first time. Supply chain "
            "disruptions caused by climate events in key growing regions pushed arabica futures to "
            "a 45-year high, affecting cafés and consumers worldwide."
        ),
        "reference_output": "The global coffee market hit $100 billion in 2025, with Vietnam becoming the top producer and arabica prices reaching a 45-year high due to climate-driven supply disruptions.",
    },
    {
        "input_text": (
            "Researchers at MIT have unveiled an AI system called AlphaFold-Evo that can predict "
            "not just protein structures but their evolutionary trajectories. The model was trained "
            "on 500 million protein sequences and can forecast how a protein might adapt over "
            "thousands of generations. This capability could accelerate drug design by identifying "
            "resistance-prone mutation pathways before they emerge in nature."
        ),
        "reference_output": "MIT's AlphaFold-Evo AI can predict protein evolutionary paths over thousands of generations, potentially accelerating drug design by anticipating resistance mutations.",
    },
    {
        "input_text": (
            "The Mars Sample Return mission faced a significant setback after NASA announced a "
            "two-year delay due to budget constraints and technical challenges with the Earth "
            "Entry Vehicle. The samples, collected by Perseverance rover since 2021, are stored "
            "in titanium tubes on the Martian surface. Scientists hope the rocks could provide "
            "evidence of ancient microbial life if the mission proceeds."
        ),
        "reference_output": "NASA delayed the Mars Sample Return mission by two years due to budget and technical issues; Perseverance samples awaiting return may hold evidence of ancient life.",
    },
]

CODE_SAMPLES = [
    {
        "input_text": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"",
        "reference_output": "for idx, elem in enumerate(numbers):\n    for idx2, elem2 in enumerate(numbers):\n        if idx != idx2:\n            distance = abs(elem - elem2)\n            if distance < threshold:\n                return True\nreturn False",
    },
    {
        "input_text": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group into separate strings and return the list of those.\n    Separate groups are balanced and not nested within each other.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"",
        "reference_output": "result = []\ncurrent_string = []\ncurrent_depth = 0\nfor c in paren_string:\n    if c == '(':\n        current_depth += 1\n        current_string.append(c)\n    elif c == ')':\n        current_depth -= 1\n        current_string.append(c)\n        if current_depth == 0:\n            result.append(''.join(current_string))\n            current_string = []\nreturn result",
    },
    {
        "input_text": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"",
        "reference_output": "return number % 1.0",
    },
    {
        "input_text": "def below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"",
        "reference_output": "balance = 0\nfor op in operations:\n    balance += op\n    if balance < 0:\n        return True\nreturn False",
    },
    {
        "input_text": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"",
        "reference_output": "mean = sum(numbers) / len(numbers)\nreturn sum(abs(x - mean) for x in numbers) / len(numbers)",
    },
]

CREATIVE_SAMPLES = [
    {
        "input_text": "A lighthouse keeper discovers that the light has been warning ships away from something other than rocks.",
        "reference_output": "For forty years, Miriam had kept the light burning without question. But on the night the fog rolled in thicker than she'd ever seen, she climbed down to the shore and found the water perfectly still — not a single wave. Whatever the light had been keeping away, it had stopped moving.",
    },
    {
        "input_text": "Two strangers are trapped in an elevator and realise they have met before in a dream.",
        "reference_output": "He recognised her before she spoke — the red scarf, the way she tilted her head when the lights flickered. 'The bridge,' he said. She went pale. 'You were standing at the end of it.' The elevator hummed between floors. Neither of them pressed a button.",
    },
    {
        "input_text": "The last bookshop on Earth is about to close its doors forever.",
        "reference_output": "The sign read CLOSING SALE — EVERYTHING MUST GO, but no one came. Elara ran her hand along the final shelf, each spine a world she'd visited a hundred times. Outside, screens glowed in every window. She locked the door slowly, placed the key on the step, and walked away without looking back.",
    },
    {
        "input_text": "A scientist invents a machine that translates the emotions of plants.",
        "reference_output": "The first thing the oak said was: patience. The second was: thirst. Dr. Chen adjusted the electrode behind its bark and waited. By nightfall, the oak was describing something that took her three weeks to translate — a memory, she eventually understood, of a storm from two hundred years ago.",
    },
    {
        "input_text": "An old map leads not to treasure but to a conversation that changes everything.",
        "reference_output": "The coordinates led to a diner outside Flagstaff. Booth seven. A woman his grandmother's age was already there, coffee in hand, as if she'd been waiting since the map was drawn. 'I wondered,' she said, 'if anyone would ever come.' He sat down. They talked until closing time.",
    },
]

DIALOGUE_SAMPLES = [
    {
        "input_text": "User: I need to find a cheap hotel in the north part of town.\nAssistant: I found a few options. Do you have a preference for the number of stars?\nUser: I'd like at least 3 stars please.",
        "reference_output": "I have a 3-star hotel called the Ashley Hotel in the north. Would you like me to book it?",
    },
    {
        "input_text": "User: Can you help me book a train from Cambridge to London?\nAssistant: Sure! What day would you like to travel?\nUser: I need to leave on Monday and arrive before 12:00.",
        "reference_output": "There is a train departing at 09:01 and arriving at 10:51. Shall I book that for you?",
    },
    {
        "input_text": "User: I'm looking for an Italian restaurant that is moderately priced.\nAssistant: I found several. Do you have a preference for area?\nUser: The city centre please.",
        "reference_output": "Prezzo is a moderately priced Italian restaurant in the city centre. Would you like their phone number?",
    },
    {
        "input_text": "User: I need a taxi to get to the train station by 3pm.\nAssistant: Where will you be departing from?\nUser: From the hotel I'm staying at, the Gonville Hotel.",
        "reference_output": "I'll book a taxi from the Gonville Hotel to the train station arriving by 3pm. Can I get your contact number?",
    },
    {
        "input_text": "User: Is there an attraction in Cambridge related to science?\nAssistant: Yes, the Whipple Museum is a great option. Is there anything specific you're interested in?\nUser: I'd like the address and opening hours.",
        "reference_output": "The Whipple Museum is on Free School Lane. Open Mon–Fri 12:30–16:30, free entry.",
    },
]

TASK_DATA = {
    "summarization":   SUMMARIZATION_SAMPLES[:SAMPLES_PER_TASK],
    "code":            CODE_SAMPLES[:SAMPLES_PER_TASK],
    "creative_writing": CREATIVE_SAMPLES[:SAMPLES_PER_TASK],
    "dialogue":        DIALOGUE_SAMPLES[:SAMPLES_PER_TASK],
}

# ── Heuristic "model" responses ───────────────────────────────────────────────

def _deterministic_hash(text: str, model: str, level: str) -> float:
    """Stable float in [0,1] from text+model+level for reproducible variation."""
    h = hashlib.md5((text + model + level).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def _extract_sentences(text: str) -> list:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if len(s.split()) >= 4]

def _words(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())

def simulate_response(input_text: str, reference: str, task: str,
                      model_name: str, variant: str, profile: dict) -> str:
    """
    Generate a heuristic response that blends input + reference sentences
    according to the model's coherence score. Higher coherence → more
    reference-aligned; lower → more paraphrased/extractive from input.
    """
    rng_seed = int(_deterministic_hash(input_text, model_name, variant) * 1e9)
    rng = np.random.default_rng(rng_seed)

    coherence    = profile["coherence"]
    min_w, max_w = profile["response_len"]
    target_words = int(rng.integers(min_w, max_w))

    if task == "code":
        # For code tasks, return a plausible code snippet
        lines = reference.strip().split("\n")
        keep  = max(1, int(len(lines) * coherence))
        snippet = "\n".join(lines[:keep])
        if len(snippet.split()) < min_w:
            snippet += "\n    pass  # implementation truncated"
        return snippet

    ref_sents   = _extract_sentences(reference)
    input_sents = _extract_sentences(input_text[:600])

    # Mix reference sentences (high coherence) with input extracts (low coherence)
    if ref_sents:
        n_ref   = max(1, round(len(ref_sents) * coherence))
        n_input = max(0, round(len(input_sents) * (1 - coherence) * 0.5))
        pool    = (rng.choice(ref_sents, size=min(n_ref, len(ref_sents)), replace=False).tolist()
                   + (rng.choice(input_sents, size=min(n_input, len(input_sents)), replace=False).tolist()
                      if input_sents else []))
        rng.shuffle(pool)
        response = " ".join(pool)
    else:
        response = reference[:300]

    # Trim to target_words
    words = response.split()
    if len(words) > target_words:
        response = " ".join(words[:target_words]) + "."

    return response.strip() or "No output generated."

def simulate_ppl(input_text: str, response: str,
                 model_name: str, variant: str, profile: dict) -> float:
    """
    Compute a heuristic perplexity:
      • Base PPL from profile, jittered by input length and variant index.
      • d3 (most explicit prompt) → slightly lower PPL than d1.
    """
    rng_seed = int(_deterministic_hash(input_text, model_name, "ppl_" + variant) * 1e9)
    rng      = np.random.default_rng(rng_seed)

    base   = profile["ppl_base"]
    spread = profile["ppl_spread"]
    # Longer inputs → higher PPL
    length_factor = min(1.5, len(input_text.split()) / 80)
    # Variant d3 is more explicit → slightly easier for model
    variant_offset = {"d1": 0.5, "d2": 0.0, "d3": -0.5}.get(variant, 0.0)

    ppl = base + length_factor * spread * 0.4 + variant_offset + rng.normal(0, spread * 0.3)
    return round(max(2.0, ppl), 4)

def simulate_bf(response: str, model_name: str, variant: str, profile: dict) -> float:
    """
    Branching factor: proxy for token-level entropy.
    More words/unique tokens → higher BF.
    """
    rng_seed = int(_deterministic_hash(response, model_name, "bf_" + variant) * 1e9)
    rng      = np.random.default_rng(rng_seed)

    words   = _words(response)
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)

    base   = profile["bf_base"]
    spread = profile["bf_spread"]
    bf     = base * 0.7 + unique_ratio * 0.3 + rng.normal(0, spread * 0.3)
    return round(float(np.clip(bf, 0.05, 0.99)), 4)

# ── Evaluation metrics ────────────────────────────────────────────────────────

def eval_rouge(hypothesis: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not hypothesis.strip() or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _ROUGE.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def eval_bleu(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU with smoothing (handles short outputs)."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH)
    return round(float(score), 4)


def eval_lex_sim(hypothesis: str, reference: str) -> float:
    """Lexical similarity via SequenceMatcher (0–1)."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    return round(difflib.SequenceMatcher(None,
                     hypothesis.lower(), reference.lower()).ratio(), 4)


def eval_exact_match(hypothesis: str, reference: str, task: str) -> int:
    """1 if outputs match exactly after normalising whitespace (mainly useful for code)."""
    h = " ".join(hypothesis.split()).strip().lower()
    r = " ".join(reference.split()).strip().lower()
    return int(h == r)


def compute_eval_metrics(response: str, reference: str, task: str) -> dict:
    """All eval metrics in one call."""
    rouge  = eval_rouge(response, reference)
    bleu   = eval_bleu(response, reference)
    lex    = eval_lex_sim(response, reference)
    em     = eval_exact_match(response, reference, task)
    return {**rouge, "bleu": bleu, "lex_sim": lex, "exact_match": em}


# ── PRI Framework — CPU-only implementations ──────────────────────────────────

def _word_freq_dist(text: str) -> np.ndarray:
    """Normalised word-frequency array (for JS-divergence)."""
    words = _words(text)
    if not words:
        return np.array([1e-10])
    freq: dict = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    total = sum(freq.values())
    return np.array(sorted([v / total for v in freq.values()]))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (numpy-only, range [0, 1])."""
    n = max(len(p), len(q))
    pp = np.pad(p, (0, n - len(p)))
    qq = np.pad(q, (0, n - len(q)))
    s  = pp.sum(); pp = pp / s  if s > 0 else pp
    s  = qq.sum(); qq = qq / s  if s > 0 else qq
    m  = 0.5 * (pp + qq)
    eps = 1e-10
    def _kl(a, b):
        return float(np.sum(a * np.log((a + eps) / (b + eps))))
    return float(np.clip(0.5 * _kl(pp, m) + 0.5 * _kl(qq, m), 0.0, 1.0))


def _tf_vector(text: str, vocab: list) -> np.ndarray:
    """Term-frequency vector over a shared vocabulary."""
    words = _words(text)
    if not words:
        return np.zeros(len(vocab))
    counts = {}
    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
    vec  = np.array([counts.get(w, 0) / len(words) for w in vocab])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _harmonic_mean(values: list) -> float:
    """Harmonic mean; skips zeros."""
    vals = [v for v in values if v > 1e-9]
    if not vals:
        return 0.0
    return round(len(vals) / sum(1.0 / v for v in vals), 4)


# ── ORI sub-metrics ───────────────────────────────────────────────────────────

def compute_sms(outputs: list) -> float:
    """
    Semantic Manifold Shift (SMS):
    Mean pairwise cosine *distance* between TF-IDF output vectors.
    Range [0, 1].  Higher → more semantic drift → less robust.
    """
    all_words = []
    for o in outputs:
        all_words.extend(_words(o))
    vocab = list(set(all_words))
    if not vocab:
        return 0.0
    vecs  = [_tf_vector(o, vocab) for o in outputs]
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(1.0 - _cosine_sim(vecs[i], vecs[j]))
    return round(float(np.mean(dists)) if dists else 0.0, 4)


def compute_auc_e(rougeL_scores: list) -> float:
    """
    AUC-E (Performance Elasticity):
    1 - CV(ROUGE-L), where CV = std / mean across K variants.
    Range [0, 1].  Higher → more stable ROUGE-L → better robustness.

    Replaces the trapezoid-AUC formulation, which collapsed to avg ROUGE-L
    (r ≈ 0.996) and provided no independent signal.
    """
    n = len(rougeL_scores)
    if n < 2:
        return float(rougeL_scores[0]) if rougeL_scores else 0.0
    mean = float(np.mean(rougeL_scores))
    if mean < 1e-9:
        return 0.0
    cv = float(np.std(rougeL_scores, ddof=1)) / mean
    return round(float(np.clip(1.0 - cv, 0.0, 1.0)), 4)


def compute_trd(outputs: list, input_text: str) -> float:
    """
    Topic Relevance Drift (TRD):
    Variance of JS-divergence between each output's word-dist and the source.
    Scaled to [0, 1] by clamping.  Lower → more faithful → better robustness.
    """
    src_dist = _word_freq_dist(input_text)
    js_vals  = [_js_divergence(_word_freq_dist(o), src_dist) for o in outputs]
    if len(js_vals) < 2:
        return 0.0
    return round(float(np.clip(np.var(js_vals) * 20.0, 0.0, 1.0)), 4)


def compute_kpig(outputs: list) -> float:
    """
    Key-Point Instruction Gain (KPIG):
    Mean pairwise TF cosine similarity across all output pairs.
    Range [0, 1].  Higher → more semantically consistent → better robustness.

    Replaces the CV-of-word-counts formulation (which was unrelated to key-point
    information and inconsistent with kpig_metric.py's semantic definition).
    """
    if len(outputs) < 2:
        return 1.0
    all_words: list = []
    for o in outputs:
        all_words.extend(_words(o))
    vocab = list(set(all_words))
    if not vocab:
        return 0.0
    vecs = [_tf_vector(o, vocab) for o in outputs]
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(_cosine_sim(vecs[i], vecs[j]))
    return round(float(np.clip(np.mean(sims), 0.0, 1.0)), 4)


def compute_ori(sms: float, auc_e: float, trd: float, kpig: float) -> dict:
    """
    Output Robustness Index  ORI = HM(norm_sms, norm_auc_e, norm_trd, norm_kpig).

    Normalisation:
      norm_sms   = 1 – sms    (less TF-drift = better)
      norm_auc_e = auc_e      (already ↑ = better; now 1-CV(rougeL))
      norm_trd   = 1 – trd    (less topic-drift = better)
      norm_kpig  = kpig       (already ↑ = better; now mean pairwise cosine sim)
    """
    norm_sms   = round(float(np.clip(1.0 - sms, 0.0, 1.0)), 4)
    norm_auc_e = round(float(np.clip(auc_e,      0.0, 1.0)), 4)
    norm_trd   = round(float(np.clip(1.0 - trd,  0.0, 1.0)), 4)
    norm_kpig  = round(float(np.clip(kpig,        0.0, 1.0)), 4)
    ori        = _harmonic_mean([norm_sms, norm_auc_e, norm_trd, norm_kpig])
    return {
        "norm_sms": norm_sms, "norm_auc_e": norm_auc_e,
        "norm_trd": norm_trd, "norm_kpig":  norm_kpig,
        "ori": ori,
    }


# ── IFI sub-metrics ───────────────────────────────────────────────────────────

def compute_ifi(ppl_vals: list, bf_vals: list, alpha: float = 0.5) -> dict:
    """
    Intrinsic Fidelity Index  IFI = HM(norm_ppl_var, norm_pc_stab).

    Normalisation:
      norm_ppl_var = exp(–α · σ_ppl / μ_ppl)   (lower CV = more stable)
      norm_pc_stab = exp(–α · σ_bf)             (lower BF variance = more stable)

    ppl_var is stored as the CV (σ/μ) clamped to [0, 1] so it is a proper
    normalised metric. Previous versions stored raw σ_ppl which could exceed 1.
    """
    ppl_arr  = np.array(ppl_vals, dtype=float)
    bf_arr   = np.array(bf_vals,  dtype=float)
    ppl_std  = float(np.std(ppl_arr))  if len(ppl_vals) > 1 else 0.0
    ppl_mean = float(np.mean(ppl_arr)) if len(ppl_vals) > 0 else 1.0
    bf_std   = float(np.std(bf_arr))   if len(bf_vals)  > 1 else 0.0

    # Normalized ppl_var: CV clamped to [0, 1]
    ppl_cv = ppl_std / max(ppl_mean, 1.0)
    ppl_var_normalized = round(float(np.clip(ppl_cv, 0.0, 1.0)), 4)

    norm_ppl_var = float(np.exp(-alpha * ppl_cv))
    norm_pc_stab = float(np.exp(-alpha * bf_std))
    ifi          = _harmonic_mean([norm_ppl_var, norm_pc_stab])
    return {
        "ppl_var":      ppl_var_normalized,          # normalized CV ∈ [0, 1]
        "pc_stab":      round(bf_std,        4),
        "norm_ppl_var": round(norm_ppl_var,  4),
        "norm_pc_stab": round(norm_pc_stab,  4),
        "ifi":          round(ifi,           4),
    }


# ── PRI composite ─────────────────────────────────────────────────────────────

def compute_pri(ori: float, ifi: float) -> float:
    """PRI = HM(ORI, IFI)."""
    return _harmonic_mean([ori, ifi])


# ── Consistency + Human Score ─────────────────────────────────────────────────

def compute_cs(outputs: list) -> float:
    """
    Consistency Score (CS):
    Mean pairwise ROUGE-L between all variant outputs.
    Higher → more consistent outputs across variants.
    """
    if len(outputs) < 2:
        return 0.0
    scores = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            scores.append(eval_rouge(outputs[i], outputs[j])["rougeL"])
    return round(float(np.mean(scores)) if scores else 0.0, 4)


def compute_hs(avg_rougeL: float, avg_bleu: float, cs: float) -> float:
    """
    Human Score heuristic (HS):
    Reference-grounded quality signals only — PRI removed to break circularity.

    Previous formula included PRI as an input (r(HS, PRI) ≈ 0.959), making HS
    nearly identical to PRI and adding no independent signal to the final score.
    """
    return round(0.40 * avg_rougeL + 0.35 * avg_bleu + 0.25 * cs, 4)


def diagnosis_label(ori: float, ifi: float, threshold: float = 0.70) -> str:
    """
    2×2 Diagnosis Matrix (ORI × IFI):
      High ORI + High IFI → True Robustness
      High ORI + Low  IFI → Evaluation Artifact
      Low  ORI + High IFI → Stochastic Luck
      Low  ORI + Low  IFI → Knowledge Boundary

    Threshold raised from 0.5 to 0.70 so that the four quadrants are actually
    populated. With threshold=0.5 every sample fell into "True Robustness".
    """
    hi_ori = ori >= threshold
    hi_ifi = ifi >= threshold
    if   hi_ori and     hi_ifi: return "True Robustness"
    elif hi_ori and not hi_ifi: return "Evaluation Artifact"
    elif not hi_ori and hi_ifi: return "Stochastic Luck"
    else:                       return "Knowledge Boundary"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total_rows = SAMPLES_PER_TASK * 4 * len(MODEL_PROFILES)
    print("=" * 70)
    print("  GenSens Lightweight Dataset Generator  (v3 — PRI Framework)")
    print(f"  Samples/task: {SAMPLES_PER_TASK}  |  Tasks: 4  |  Variants: v1…v5")
    print(f"  Models: {len(MODEL_PROFILES)} (all < 2B params)")
    print(f"  Output: {total_rows} rows  (wide format + PRI / ORI / IFI columns)")
    print("=" * 70)
    print()

    all_records = []

    # ── Step 1: Build skeleton records — one per (sample × model) ────────────
    for task, samples in TASK_DATA.items():
        for s_idx, sample in enumerate(samples):
            variants = build_variants(sample["input_text"], task)
            for model_name, profile in MODEL_PROFILES.items():
                rec = {
                    "topic_label":      task,
                    "sample_idx":       s_idx,
                    "model":            model_name,
                    "input_text":       sample["input_text"],
                    "original_prompt":  ORIGINAL_PROMPT[task],
                    "reference_output": sample["reference_output"],
                    "_variants":        variants,
                    "_profile":         profile,
                }
                all_records.append(rec)

    # ── Step 2: Fill per-variant responses + basic eval metrics ──────────────
    print("── Step 1/3  Generating responses + per-variant eval metrics ──")
    for rec in all_records:
        task      = rec["topic_label"]
        model     = rec["model"]
        profile   = rec["_profile"]
        variants  = rec["_variants"]
        reference = rec["reference_output"]

        for v in VARIANTS:
            vi     = v[1]               # "1"…"5"
            prompt = variants[v]
            resp   = simulate_response(rec["input_text"], reference,
                                       task, model, v, profile)
            ppl    = simulate_ppl(rec["input_text"], resp, model, v, profile)
            bf     = simulate_bf(resp, model, v, profile)
            evals  = compute_eval_metrics(resp, reference, task)

            rec[f"prompt_variant_{vi}"] = prompt[:250].replace("\n", " ")
            rec[f"model_output_v{vi}"]  = resp[:300].replace("\n", " ")
            rec[f"ppl_v{vi}"]           = ppl
            rec[f"bf_v{vi}"]            = bf
            rec[f"rouge1_v{vi}"]        = evals["rouge1"]
            rec[f"rouge2_v{vi}"]        = evals["rouge2"]
            rec[f"rougeL_v{vi}"]        = evals["rougeL"]
            rec[f"bleu_v{vi}"]          = evals["bleu"]
            rec[f"lex_sim_v{vi}"]       = evals["lex_sim"]
            rec[f"exact_match_v{vi}"]   = evals["exact_match"]

    # ── Clean internal keys ───────────────────────────────────────────────────
    for rec in all_records:
        del rec["_variants"]
        del rec["_profile"]

    # ── Step 3: Compute PRI framework metrics (row-level) ─────────────────────
    print("── Step 2/3  Computing PRI / ORI / IFI framework metrics ──")
    N = range(1, 6)
    for rec in all_records:
        outputs       = [rec.get(f"model_output_v{i}", "") for i in N]
        rougeL_scores = [rec.get(f"rougeL_v{i}", 0.0)     for i in N]
        ppl_vals      = [rec.get(f"ppl_v{i}",    0.0)     for i in N]
        bf_vals       = [rec.get(f"bf_v{i}",     0.0)     for i in N]
        bleu_scores   = [rec.get(f"bleu_v{i}",   0.0)     for i in N]

        # ORI sub-metrics
        sms   = compute_sms(outputs)
        auc_e = compute_auc_e(rougeL_scores)
        trd   = compute_trd(outputs, rec["input_text"])
        kpig  = compute_kpig(outputs)

        # IFI sub-metrics
        ifi_r = compute_ifi(ppl_vals, bf_vals)

        # Composite indices
        ori_r = compute_ori(sms, auc_e, trd, kpig)  # alpha arg removed; kpig now ↑=better
        pri   = compute_pri(ori_r["ori"], ifi_r["ifi"])

        # Quality / consistency
        cs    = compute_cs(outputs)
        avg_rL  = float(np.mean(rougeL_scores))
        avg_bl  = float(np.mean(bleu_scores))
        hs    = compute_hs(avg_rL, avg_bl, cs)
        diag  = diagnosis_label(ori_r["ori"], ifi_r["ifi"])

        # Store
        rec["sms"]     = sms
        rec["auc_e"]   = auc_e
        rec["trd"]     = trd
        rec["kpig"]    = kpig
        rec["ppl_var"] = ifi_r["ppl_var"]
        rec["pc_stab"] = ifi_r["pc_stab"]
        rec["ifi"]     = ifi_r["ifi"]
        rec["ori"]     = ori_r["ori"]
        rec["pri"]     = pri
        rec["cs"]      = cs
        rec["hs"]      = hs
        rec["diagnosis"] = diag

    # ── Save JSON + CSV per task + combined ───────────────────────────────────
    print("── Step 3/3  Saving files ──")
    for task in TASK_DATA:
        task_recs = [r for r in all_records if r["topic_label"] == task]
        _save_json(task_recs, DATA_DIR / f"dataset_{task}.json")
        _save_wide_csv(task_recs, DATA_DIR / f"dataset_{task}.csv")

    _save_json(all_records, DATA_DIR / "dataset_all_tasks.json")
    _save_wide_csv(all_records, DATA_DIR / "dataset_all_tasks.csv")

    # ── Print results summary ─────────────────────────────────────────────────
    _print_results(all_records)
    # Count columns from last saved CSV
    import csv as _csv
    with open(DATA_DIR / "dataset_all_tasks.csv", newline="") as f:
        ncols = len(next(_csv.reader(f)))
    print(f"\n[done]  {len(all_records)} rows × {ncols} cols → {DATA_DIR}")
    print("  Open data/dataset_all_tasks.csv to inspect the full output.")


def _print_results(records: list):
    """Pretty-print evaluation results (including PRI framework) to terminal."""
    import os
    try:    W = min(os.get_terminal_size().columns, 110)
    except: W = 110

    def _mean(vals): return sum(vals) / len(vals) if vals else 0.0
    N = range(1, 6)

    # ── Per-model summary: quality metrics ────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  EVALUATION RESULTS SUMMARY  (v3 — PRI Framework)")
    print(f"{'═'*W}")

    model_stats: dict = {}
    for rec in records:
        m = rec["model"].split("/")[-1]
        if m not in model_stats:
            model_stats[m] = {k: [] for k in
                              ["ppl","bf","rouge1","rouge2","rougeL",
                               "bleu","lex_sim","ppl_std",
                               "pri","ori","ifi","cs","hs","sms","auc_e","trd","kpig"]}
        ms = model_stats[m]
        ms["ppl"].append(_mean([rec.get(f"ppl_v{i}",0)     for i in N]))
        ms["bf"].append (_mean([rec.get(f"bf_v{i}",0)      for i in N]))
        ms["rouge1"].append(_mean([rec.get(f"rouge1_v{i}",0) for i in N]))
        ms["rouge2"].append(_mean([rec.get(f"rouge2_v{i}",0) for i in N]))
        ms["rougeL"].append(_mean([rec.get(f"rougeL_v{i}",0) for i in N]))
        ms["bleu"].append  (_mean([rec.get(f"bleu_v{i}",0)   for i in N]))
        ms["lex_sim"].append(_mean([rec.get(f"lex_sim_v{i}",0) for i in N]))
        ppls = [rec.get(f"ppl_v{i}",0) for i in N]
        mu   = _mean(ppls)
        ms["ppl_std"].append((_mean([(v-mu)**2 for v in ppls]))**0.5)
        # PRI metrics
        for k in ["pri","ori","ifi","cs","hs","sms","auc_e","trd","kpig"]:
            ms[k].append(rec.get(k, 0.0))

    # Table 1 — quality
    print(f"\n  TABLE 1 — Per-Model Quality Metrics")
    print(f"\n  {'Model':<28} {'PPL↓':>7} {'BF↑':>7} {'ROUGE-1↑':>9} {'ROUGE-2↑':>9} "
          f"{'ROUGE-L↑':>9} {'BLEU↑':>7} {'LexSim↑':>8} {'PPL-σ↓':>8}")
    print(f"  {'─'*28} {'─'*7} {'─'*7} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*8} {'─'*8}")
    for m, s in model_stats.items():
        print(f"  {m:<28} {_mean(s['ppl']):>7.3f} {_mean(s['bf']):>7.3f} "
              f"{_mean(s['rouge1']):>9.4f} {_mean(s['rouge2']):>9.4f} "
              f"{_mean(s['rougeL']):>9.4f} {_mean(s['bleu']):>7.4f} "
              f"{_mean(s['lex_sim']):>8.4f} {_mean(s['ppl_std']):>8.4f}")

    # Table 2 — PRI framework
    print(f"\n{'─'*W}")
    print(f"  TABLE 2 — PRI Framework  (Prompt Robustness Index)")
    print(f"\n  {'Model':<28} {'PRI↑':>6} {'ORI↑':>6} {'IFI↑':>6} "
          f"{'SMS↓':>6} {'AUC-E↑':>7} {'TRD↓':>6} {'KPIG↓':>7} "
          f"{'CS↑':>6} {'HS↑':>6}")
    print(f"  {'─'*28} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")
    for m, s in model_stats.items():
        print(f"  {m:<28} {_mean(s['pri']):>6.4f} {_mean(s['ori']):>6.4f} "
              f"{_mean(s['ifi']):>6.4f} {_mean(s['sms']):>6.4f} "
              f"{_mean(s['auc_e']):>7.4f} {_mean(s['trd']):>6.4f} "
              f"{_mean(s['kpig']):>7.4f} {_mean(s['cs']):>6.4f} "
              f"{_mean(s['hs']):>6.4f}")

    # Diagnosis distribution
    print(f"\n{'─'*W}")
    print(f"  TABLE 3 — Diagnosis Labels  (ORI × IFI  2×2 Matrix)")
    from collections import Counter
    diag_counts: dict = {}
    for rec in records:
        m = rec["model"].split("/")[-1]
        if m not in diag_counts:
            diag_counts[m] = Counter()
        diag_counts[m][rec.get("diagnosis", "?")] += 1
    labels = ["True Robustness", "Evaluation Artifact", "Stochastic Luck", "Knowledge Boundary"]
    header = f"  {'Model':<28}" + "".join(f" {lb[:16]:>17}" for lb in labels)
    print(f"\n{header}")
    print(f"  {'─'*28}" + "".join(f" {'─'*16:>17}" for _ in labels))
    for m, ctr in diag_counts.items():
        row = f"  {m:<28}"
        for lb in labels:
            row += f" {ctr.get(lb, 0):>17d}"
        print(row)

    # Table 4 — per-task breakdown
    print(f"\n{'─'*W}")
    print(f"  TABLE 4 — Per-Task Breakdown  (averaged across all models)")
    tasks_list = ["summarization","code","creative_writing","dialogue"]
    print(f"\n  {'Task':<20} {'ROUGE-L↑':>9} {'BLEU↑':>8} {'PPL-σ↓':>8} "
          f"{'PRI↑':>6} {'ORI↑':>6} {'IFI↑':>6} {'CS↑':>6}")
    print(f"  {'─'*20} {'─'*9} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for t in tasks_list:
        recs = [r for r in records if r["topic_label"] == t]
        if not recs: continue
        rLs   = [_mean([r.get(f"rougeL_v{i}",0) for i in N]) for r in recs]
        bleus = [_mean([r.get(f"bleu_v{i}",0)   for i in N]) for r in recs]
        stds  = []
        for r in recs:
            ps = [r.get(f"ppl_v{i}",0) for i in N]
            mu = _mean(ps); stds.append((_mean([(v-mu)**2 for v in ps]))**0.5)
        pris  = [r.get("pri",0) for r in recs]
        oris  = [r.get("ori",0) for r in recs]
        ifis  = [r.get("ifi",0) for r in recs]
        css   = [r.get("cs",0)  for r in recs]
        print(f"  {t:<20} {_mean(rLs):>9.4f} {_mean(bleus):>8.4f} {_mean(stds):>8.4f} "
              f"{_mean(pris):>6.4f} {_mean(oris):>6.4f} {_mean(ifis):>6.4f} "
              f"{_mean(css):>6.4f}")

    # Table 5 — prompt sensitivity
    print(f"\n{'─'*W}")
    print(f"  TABLE 5 — Prompt Sensitivity  (avg ROUGE-L per variant)")
    print(f"\n  {'Variant':<12} {'Avg ROUGE-L':>12} {'Avg BLEU':>10} {'Avg PPL':>9}  Bar")
    print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*9}  {'─'*32}")
    for i in N:
        rL   = _mean([r.get(f"rougeL_v{i}",0) for r in records])
        bleu = _mean([r.get(f"bleu_v{i}",0)   for r in records])
        ppl  = _mean([r.get(f"ppl_v{i}",0)    for r in records])
        bar  = "█" * int(rL * 30)
        print(f"  v{i} (variant {i})  {rL:>12.4f} {bleu:>10.4f} {ppl:>9.3f}  {bar}")

    print(f"\n{'═'*W}")


def _save_json(records: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  [json] {len(records)} records → {path.name}")


def _save_wide_csv(records: list, path: Path):
    """
    Wide format with eval + PRI metrics: one row per (sample × model).
    Column order:
      topic_label | sample_idx | model |
      input_text | original_prompt |
      prompt_variant_1…5 |
      model_output_v1…5 |
      ppl_v1…5 | bf_v1…5 |
      rouge1_v1…5 | rouge2_v1…5 | rougeL_v1…5 |
      bleu_v1…5 | lex_sim_v1…5 | exact_match_v1…5 |
      avg_ppl | avg_bf | ppl_std |
      avg_rouge1 | avg_rouge2 | avg_rougeL | avg_bleu | avg_lex_sim |
      rougeL_std | bleu_std |
      ── PRI Framework ──
      sms | auc_e | trd | kpig |
      ppl_var | pc_stab |
      ori | ifi | pri |
      cs | hs | diagnosis |
      reference_output
    """
    N = range(1, 6)
    col_order = (
        ["topic_label", "sample_idx", "model", "input_text", "original_prompt"]
        + [f"prompt_variant_{i}" for i in N]
        + [f"model_output_v{i}"  for i in N]
        + [f"ppl_v{i}"           for i in N]
        + [f"bf_v{i}"            for i in N]
        + [f"rouge1_v{i}"        for i in N]
        + [f"rouge2_v{i}"        for i in N]
        + [f"rougeL_v{i}"        for i in N]
        + [f"bleu_v{i}"          for i in N]
        + [f"lex_sim_v{i}"       for i in N]
        + [f"exact_match_v{i}"   for i in N]
        + ["avg_ppl", "avg_bf", "ppl_std",
           "avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_bleu", "avg_lex_sim",
           "rougeL_std", "bleu_std"]
        # PRI Framework columns
        + ["sms", "auc_e", "trd", "kpig",
           "ppl_var", "pc_stab",
           "ori", "ifi", "pri",
           "cs", "hs", "diagnosis"]
        + ["reference_output"]
    )

    def _mean(vals): return round(sum(vals)/len(vals), 4) if vals else 0.0
    def _std(vals):
        if len(vals) < 2: return 0.0
        m = _mean(vals)
        return round((sum((v-m)**2 for v in vals)/len(vals))**0.5, 4)

    rows = []
    for rec in records:
        row = {c: rec.get(c, "") for c in col_order}
        # Trim text
        for col in ["input_text", "reference_output"]:
            if isinstance(row[col], str):
                row[col] = row[col][:300].replace("\n", " ")
        # Aggregate summary columns
        ppls  = [rec.get(f"ppl_v{i}",    0) for i in N]
        bfs   = [rec.get(f"bf_v{i}",     0) for i in N]
        r1s   = [rec.get(f"rouge1_v{i}", 0) for i in N]
        r2s   = [rec.get(f"rouge2_v{i}", 0) for i in N]
        rLs   = [rec.get(f"rougeL_v{i}", 0) for i in N]
        bleus = [rec.get(f"bleu_v{i}",   0) for i in N]
        lexs  = [rec.get(f"lex_sim_v{i}",0) for i in N]
        row["avg_ppl"]     = _mean(ppls)
        row["avg_bf"]      = _mean(bfs)
        row["ppl_std"]     = _std(ppls)
        row["avg_rouge1"]  = _mean(r1s)
        row["avg_rouge2"]  = _mean(r2s)
        row["avg_rougeL"]  = _mean(rLs)
        row["avg_bleu"]    = _mean(bleus)
        row["avg_lex_sim"] = _mean(lexs)
        row["rougeL_std"]  = _std(rLs)
        row["bleu_std"]    = _std(bleus)
        rows.append(row)

    df = pd.DataFrame(rows, columns=col_order)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  [csv]  {len(rows)} rows × {len(col_order)} cols → {path.name}")


if __name__ == "__main__":
    main()
