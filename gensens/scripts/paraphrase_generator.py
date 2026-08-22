"""
GenSens Paraphrase Generator
=============================
Supports two generation backends (both GPU-only, H100/A100):

  "vllm"  (recommended) → vLLM batched engine, 70B-AWQ or any HF model
  "llama"               → HF transformers Llama 8B, single-sequence

Both backends share:
  • 16 diverse strategy instructions (per-task whitelist applied)
  • Task-specific system prompts
  • SBERT "all-mpnet-base-v2" similarity filter ≥ 0.82
  • Deduplication by first-8-word signature
  • Up to 3 retries if < n_variants valid paraphrases found
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Task-specific system prompts  (used by LLaMA/vLLM chat template)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "summarization": (
        "You are a prompt rewriting assistant. Rephrase summarization instructions "
        "while keeping the exact same meaning and intent. Do NOT change what is being "
        "asked — only change the wording. Return ONLY the rephrased instruction, nothing else."
    ),  # kept for backward compatibility; pool-aware routing uses SUMM_SYSTEM_PROMPTS
    "creative": (
        "You are a text rewriting assistant. Rephrase the following news summary/highlights "
        "while keeping all the same facts and information intact. Do NOT add, remove, or alter "
        "any facts — only change the wording and phrasing. Return ONLY the rephrased summary, "
        "nothing else."
    ),
    "dialogue": (
        "You are a question rewriting assistant. Rephrase the reading-comprehension question "
        "while keeping the exact same intent and the exact same information being requested. "
        "Do NOT change what is being asked — only change the wording. Return ONLY the "
        "rephrased question, nothing else."
    ),
    "qa": (
        "You are a question rewriting assistant. Rephrase the question while keeping the "
        "exact same information need and intent, so the same answer would still be correct. "
        "Do NOT change what is being asked — only change the wording. Return ONLY the "
        "rephrased question, nothing else."
    ),
}

# Pool-aware system prompts for summarization (Pool A = open-style, Pool B = constraint-bearing)
SUMM_SYSTEM_PROMPTS: Dict[str, str] = {
    "A": (
        "You are a prompt rewriting assistant. Your job is to rephrase "
        "summarization instructions while keeping the exact same meaning "
        "and intent. Do NOT change what is being asked — only change how "
        "it is worded. Do NOT add or remove any constraints. "
        "Return ONLY the rephrased instruction, nothing else."
    ),
    "B": (
        "You are a prompt rewriting assistant. Your job is to rephrase "
        "summarization instructions that contain a specific constraint "
        "(banned words, thematic focus, tone requirement, persona, or "
        "output structure). Rephrase the instruction while keeping the "
        "EXACT SAME constraint fully intact. The constraint must be "
        "clearly present and enforceable in the rephrased version. If "
        "the original bans certain words, the rephrase must still ban "
        "them. If it requires a specific tone, the rephrase must require "
        "the same tone. Do NOT weaken, generalize, or drop the "
        "constraint. Return ONLY the rephrased instruction, nothing else."
    ),
}

# ─────────────────────────────────────────────────────────────
# 16 paraphrase strategy instructions
# ─────────────────────────────────────────────────────────────

STRATEGY_INSTRUCTIONS: List[str] = [
    "Rewrite the following in a more formal, professional tone while preserving the exact meaning.",
    "Rewrite the following in a casual, conversational tone while preserving the exact meaning.",
    "Rewrite the following by reordering the clauses or phrases, while preserving the exact meaning.",
    "Rewrite the following by replacing key words with appropriate synonyms, while preserving the exact meaning.",
    "Rewrite the following more concisely — use fewer words while preserving the exact meaning.",
    "Rewrite the following using a completely different sentence structure while preserving the exact meaning.",
    "Rewrite the following so it begins with a completely different opening word or phrase, while preserving the exact meaning.",
    "Rewrite the following by adding a brief role or context prefix (e.g., 'As a reader, ...'), while preserving the exact meaning.",
    "Rewrite the following using passive voice constructions where possible, while preserving the exact meaning.",
    "Rewrite the following as a question or request form while preserving the exact meaning.",
    "Rewrite the following as a direct imperative command while preserving the exact meaning.",
    "Rewrite the following with slightly more detail or elaboration while preserving the exact meaning.",
    "Rewrite the following using more technical or precise vocabulary while preserving the exact meaning.",
    "Rewrite the following using simpler, everyday vocabulary while preserving the exact meaning.",
    "Rewrite the following by splitting it into shorter sentences if possible, while preserving the exact meaning.",
    "Rewrite the following by merging phrases into longer, flowing sentences while preserving the exact meaning.",
    # ── FORMAT axis (surface form of the REQUEST, not the requested output) ──
    # These deliberately keep the wording and meaning and change only the
    # surface presentation. They are near-copies by design; the format axis
    # is exempted from the similarity-ceiling / token-overlap gates below.
    "Rewrite the following changing ONLY surface formatting — capitalization, punctuation, and spacing (e.g., alter casing, swap or add dashes/colons/em-dashes) — keeping every word and the exact meaning the same.",
    "Rewrite the following by converting it into a bulleted or numbered list style (or, if it is already a list, into flowing prose), keeping the same words and exact meaning as closely as possible.",
    "Rewrite the following by changing separators and layout — add line breaks, a leading label such as 'Task:' or 'Instruction:', or restyle delimiters — while keeping the same words and the exact meaning.",
]

STRATEGY_NAMES: List[str] = [
    "formal_tone", "casual_tone", "reordered_clauses", "synonyms",
    "concise", "different_structure", "different_opening", "role_prefix",
    "passive_voice", "question_form", "imperative", "elaborated",
    "technical_vocab", "simple_vocab", "split_sentences", "merged_sentences",
    "reformat_surface", "bulletize", "separator_style",
]

assert len(STRATEGY_INSTRUCTIONS) == len(STRATEGY_NAMES) == 19

# ─────────────────────────────────────────────────────────────
# Strategy families (PUBLICATION_SPEC §6.6)
# Group the 16 ad-hoc strategies into 4 functional families and
# cap per-family rather than per-strategy. Without this, a single
# family (e.g. lexical substitution) could supply 5 of K=8 variants
# and dominate the surface-form distribution. Family caps enforce
# methodological balance reviewers expect from a "principled
# strategy taxonomy" claim.
# ─────────────────────────────────────────────────────────────

STRATEGY_FAMILY: Dict[str, str] = {
    # Lexical-substitution family: word-level changes preserving structure
    "formal_tone":         "lexical",
    "casual_tone":         "lexical",
    "synonyms":            "lexical",
    "technical_vocab":     "lexical",
    "simple_vocab":        "lexical",
    # Syntactic-rearrangement family: structural changes preserving lexis
    "reordered_clauses":   "syntactic",
    "different_structure": "syntactic",
    "different_opening":   "syntactic",
    "passive_voice":       "syntactic",
    "split_sentences":     "syntactic",
    "merged_sentences":    "syntactic",
    # Pragmatic family: speech-act modulation
    "question_form":       "pragmatic",
    "imperative":          "pragmatic",
    "role_prefix":         "pragmatic",
    # Length-manipulation family
    "concise":             "length",
    "elaborated":          "length",
    # Format family: surface form of the request (punctuation/caps/layout)
    "reformat_surface":    "format",
    "bulletize":           "format",
    "separator_style":     "format",
}
assert set(STRATEGY_FAMILY) == set(STRATEGY_NAMES), \
    "STRATEGY_FAMILY must cover all STRATEGY_NAMES"

# Balanced 5-axis budget (method.md 2026-08-22): 2 variants per axis =
# 10 variants/seed. This replaces the earlier unbalanced 2+2+1+1 (=6) scheme
# and adds the FORMAT axis, so per-axis sensitivity is measured on equal n
# across all four tasks. The `back_translation` family remains a separate
# post-hoc augmentation pass (`back_translation_family.py`, PUBLICATION_SPEC
# §7.2); its cap is registered here only so per-family logic does not
# double-count it — it is NOT part of the 10-variant balanced budget.
MAX_PER_FAMILY: Dict[str, int] = {
    "lexical":          2,
    "syntactic":        2,
    "pragmatic":        2,
    "length":           2,
    "format":           2,
    "back_translation": 2,
}

# Axes exempt from the similarity-ceiling and token-overlap diversity gates.
# FORMAT perturbations (caps/punctuation/layout) are near-copies BY DESIGN —
# the whole point is that a trivial surface change still perturbs output — so
# the standard diversity gates would reject every candidate. They still must
# pass the LOWER SBERT bound and bidirectional NLI (meaning preserved).
DIVERSITY_EXEMPT_FAMILIES: set = {"format"}

# ─────────────────────────────────────────────────────────────
# Per-task strategy whitelist
# ─────────────────────────────────────────────────────────────

STRATEGY_BY_TASK: Dict[str, List[str]] = {
    "summarization": [
        "formal_tone", "casual_tone", "reordered_clauses", "synonyms",
        "concise", "different_structure", "different_opening", "role_prefix",
        "passive_voice", "imperative", "elaborated",
        "technical_vocab", "simple_vocab", "split_sentences", "merged_sentences",
        "reformat_surface", "bulletize", "separator_style",
    ],
    "creative": [
        "formal_tone", "casual_tone", "reordered_clauses", "synonyms",
        "different_structure", "different_opening",
        "passive_voice", "elaborated",
        "technical_vocab", "simple_vocab", "split_sentences", "merged_sentences",
        "reformat_surface", "bulletize", "separator_style",
    ],
    "dialogue": [
        "formal_tone", "casual_tone", "reordered_clauses", "synonyms",
        "different_structure", "different_opening",
        "elaborated", "technical_vocab", "simple_vocab",
        "reformat_surface", "bulletize", "separator_style",
    ],
    "qa": [
        "formal_tone", "casual_tone", "reordered_clauses", "synonyms",
        "different_structure", "different_opening",
        "elaborated", "technical_vocab", "simple_vocab",
        "reformat_surface", "bulletize", "separator_style",
    ],
}


# ─────────────────────────────────────────────────────────────
# Bidirectional NLI gate (PUBLICATION_SPEC §6.1)
#
# Lazy-loaded cross-encoder NLI model used to gate paraphrase
# candidates on bidirectional entailment. Previously this gate
# ran only on QA/dialogue; per spec §6.1 we now apply it to all
# four tasks because SBERT cosine alone cannot distinguish
# paraphrases from high-overlap non-paraphrases (PAWS, Zhang et
# al. NAACL 2019).
#
# Threshold τ_NLI = 0.50 in BOTH directions. Multi-NLI ensemble
# (Tier-2 §7.1) is a follow-up that wraps `_get_nli_models()` and
# requires ≥2 of 3 votes. For Tier-1 the single primary model
# (cross-encoder/nli-deberta-v3-small) is used.
# ─────────────────────────────────────────────────────────────

_PRIMARY_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_NLI_ENTAIL_THRESHOLD_DEFAULT = 0.50
# Module-level singleton cache so we don't re-load the NLI model
# per (instance, candidate). Keyed by model id.
_NLI_MODEL_CACHE: Dict[str, Any] = {}

# PUBLICATION_SPEC §7.1 — multi-NLI ensemble (3 models, 2-of-3 majority).
#
# The single-model gate (~140M deberta-v3-small) is fast but has a
# non-trivial false-positive rate on adversarial pairs (PAWS shows ~5–10%).
# A 2-of-3 majority across three independently-trained NLI checkpoints
# pushes the compound false-positive rate below 1% at modest extra cost
# (the additional models load once and run batched).
#
# Member checkpoints:
#   - cross-encoder/nli-deberta-v3-small (primary; same one used at Tier-1)
#   - cross-encoder/nli-deberta-v3-base
#   - cross-encoder/nli-roberta-base
#
# We use cross-encoder variants for all three so the call signature is
# uniform (`model.predict([(premise, hypothesis)])` → 3-logit array).
# Mixing in the larger `roberta-large-mnli` / `bart-large-mnli`
# Seq-classifier checkpoints requires a different inference path; the
# follow-up `experiment_nli_ensemble_eval.py` script benchmarks both
# configurations on the PAWS-X test set to pick the optimal trio.
_NLI_ENSEMBLE_MODELS: List[str] = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-roberta-base",
]


def _get_nli_model(model_name: str = _PRIMARY_NLI_MODEL):
    """Lazy-load a cross-encoder NLI model (cached singleton).

    Returns None on load failure so the caller can downgrade to
    SBERT-only filtering with a warning rather than crash the
    whole generation run.
    """
    if model_name in _NLI_MODEL_CACHE:
        return _NLI_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading NLI model: {model_name}")
        m = CrossEncoder(model_name)
        _NLI_MODEL_CACHE[model_name] = m
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Failed to load NLI model {model_name}: {e}. "
            "Falling back to SBERT-only filtering for affected tasks."
        )
        _NLI_MODEL_CACHE[model_name] = None
        return None


def _get_nli_ensemble(
    model_names: Optional[List[str]] = None,
) -> List[Tuple[str, Any]]:
    """Lazy-load the NLI ensemble (cached singleton per model id).

    Returns a list of (model_name, model_or_None) pairs preserving the
    input order. A None entry means that checkpoint failed to load — the
    caller treats it as an abstain (does not count toward the majority).
    Used by `_nli_ensemble_entail` for §7.1 majority voting.
    """
    names = list(model_names) if model_names else list(_NLI_ENSEMBLE_MODELS)
    loaded: List[Tuple[str, Any]] = []
    for name in names:
        loaded.append((name, _get_nli_model(name)))
    n_ok = sum(1 for _, m in loaded if m is not None)
    if n_ok == 0:
        logger.warning(
            "[ensemble] No NLI models loaded successfully. The ensemble gate "
            "will downgrade to graceful pass-through (variants will be marked "
            "nli_passed=False but not rejected)."
        )
    else:
        logger.info(f"[ensemble] {n_ok}/{len(names)} NLI checkpoints ready.")
    return loaded


def _nli_ensemble_entail(
    ensemble: List[Tuple[str, Any]],
    premise: str,
    hypothesis: str,
    threshold: float,
    majority: int,
) -> Dict[str, Any]:
    """Run all loaded NLI checkpoints on (premise → hypothesis); majority vote.

    Returns:
      {
        "votes":      int   — number of ensemble members that asserted entail
        "n_models":   int   — number of loaded (non-None) members
        "passes":     bool  — votes >= majority
        "per_model":  Dict[str, Optional[float]] — P(entail) per checkpoint
      }

    Abstentions (None per-model) do NOT count toward the threshold; a
    pathological case of all-None returns passes=False with votes=0.
    """
    per_model: Dict[str, Optional[float]] = {}
    votes = 0
    n_models = 0
    for name, model in ensemble:
        if model is None:
            per_model[name] = None
            continue
        n_models += 1
        prob = _nli_entail_prob(model, premise, hypothesis)
        per_model[name] = round(prob, 4) if prob is not None else None
        if prob is not None and prob >= threshold:
            votes += 1
    passes = votes >= majority
    return {
        "votes":    votes,
        "n_models": n_models,
        "passes":   passes,
        "per_model": per_model,
    }


def _nli_entail_prob(model, premise: str, hypothesis: str) -> Optional[float]:
    """Run one NLI prediction premise→hypothesis; return P(entailment).

    cross-encoder/nli-deberta-v3-small returns 3 logits ordered
    [contradiction, entailment, neutral]. We softmax → index 1.
    Returns None when the model is unavailable so the caller can
    apply graceful degradation.
    """
    if model is None:
        return None
    try:
        scores = model.predict([(premise, hypothesis)])
        s = np.asarray(scores).reshape(-1)
        if s.size == 3:
            ex = np.exp(s - np.max(s))
            sm = ex / np.sum(ex)
            return float(sm[1])
        # Some checkpoints emit a single logit — interpret as raw entail prob.
        return float(s[0])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"NLI predict failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ParaphraseGenerator
# ─────────────────────────────────────────────────────────────

class ParaphraseGenerator:
    """
    Generates paraphrase variants using vLLM (recommended, batched) or
    HF-transformers LLaMA (single-sequence). Both backends are GPU-only.

    Args:
        model_backend : "vllm"  → batched vLLM engine (H100/A100, recommended)
                        "llama" → HF transformers LLaMA 8B (single-seq, slower)
        sbert_model   : SBERT model for similarity filtering (default: all-mpnet-base-v2)
        similarity_threshold : minimum SBERT cos-sim to keep a candidate (default: 0.82)
        min_word_count       : minimum words in a valid paraphrase (default: 5)
    """

    LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    VLLM_MODEL_ID  = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

    # Flaw §6.3 — per-task SBERT threshold overrides. Dialogue prompts are
    # short and high-density: with the global 0.82 threshold only ~2.8 of N
    # candidate variants survive per instance. Loosening to 0.78 for
    # dialogue restores variant yield without polluting other tasks.
    DEFAULT_TASK_THRESHOLDS: Dict[str, float] = {
        "dialogue": 0.78,
    }

    def __init__(
        self,
        model_backend:          str   = "vllm",
        sbert_model:            str   = "all-mpnet-base-v2",
        similarity_threshold:   float = 0.82,
        min_word_count:         int   = 5,
        similarity_upper:       float = 0.98,
        max_token_overlap:      float = 0.85,
        max_per_strategy:       int   = 2,
        model_id:               Optional[str] = None,
        quantization:           Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len:          int   = 4096,
        tensor_parallel_size:   int   = 1,
        task_thresholds:        Optional[Dict[str, float]] = None,
        # PUBLICATION_SPEC Tier-1 additions ────────────────────────────────
        # §6.1 — bidirectional NLI gate on ALL 4 tasks (previously QA/dialogue
        # only). Set enable_nli_gate=False only for legacy reproductions.
        enable_nli_gate:        bool  = True,
        nli_entail_threshold:   float = _NLI_ENTAIL_THRESHOLD_DEFAULT,
        nli_model_name:         str   = _PRIMARY_NLI_MODEL,
        # §7.1 — multi-NLI ensemble (3 models, 2-of-3 majority).
        # When enable_nli_ensemble=True we run each candidate through
        # ALL `nli_ensemble_models` and accept only when the bidirectional
        # entailment vote count ≥ `nli_ensemble_majority` IN BOTH DIRECTIONS.
        # Default off so existing tests / pre-§7.1 reproductions remain
        # bit-stable; the GenSens production runner sets it True via CLI.
        enable_nli_ensemble:    bool  = False,
        nli_ensemble_models:    Optional[List[str]] = None,
        nli_ensemble_majority:  int   = 2,
        # §6.2 — length-ratio filter (0.7×–1.5× of base word count).
        length_ratio_min:       float = 0.70,
        length_ratio_max:       float = 1.50,
        # §6.3 — SBERT semantic dedup vs already-accepted variants.
        semantic_dedup_threshold: float = 0.95,
        # §6.5 — best-of-N per strategy (n=3 by default). Setting n=1
        # recovers the pre-spec single-shot behaviour.
        best_of_n:              int   = 3,
        best_of_n_temperature:  float = 0.95,
        best_of_n_top_p:        float = 0.92,
        # §6.6 — strategy-family budget overrides {family: cap}. None →
        # defaults from MAX_PER_FAMILY above.
        max_per_family:         Optional[Dict[str, int]] = None,
    ):
        assert model_backend in ("llama", "vllm"), \
            "model_backend must be 'vllm' or 'llama' (GPU-only backends)"

        self.backend                = model_backend
        self.similarity_threshold   = similarity_threshold
        self.min_word_count         = min_word_count
        self.similarity_upper       = similarity_upper
        self.max_token_overlap      = max_token_overlap
        self.max_per_strategy       = max_per_strategy
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len          = max_model_len
        # vLLM tensor parallelism — set ≥2 to shard 70B AWQ across multiple
        # GPUs (e.g. A30 × 2 or 2× consumer cards). Single-card setups keep =1.
        self.tensor_parallel_size   = int(tensor_parallel_size)
        # Per-task overrides merged on top of defaults; callers can pass
        # `task_thresholds={"dialogue": 0.75, "qa": 0.84}` to fine-tune.
        self.task_thresholds        = dict(self.DEFAULT_TASK_THRESHOLDS)
        if task_thresholds:
            self.task_thresholds.update(task_thresholds)

        # PUBLICATION_SPEC Tier-1 state ────────────────────────────────────
        self.enable_nli_gate         = bool(enable_nli_gate)
        self.nli_entail_threshold    = float(nli_entail_threshold)
        self.nli_model_name          = nli_model_name
        self._nli_model              = None  # lazy-loaded on first use
        # §7.1 — ensemble state. Loaded lazily on first NLI call when enabled.
        self.enable_nli_ensemble     = bool(enable_nli_ensemble)
        self.nli_ensemble_models     = (
            list(nli_ensemble_models) if nli_ensemble_models
            else list(_NLI_ENSEMBLE_MODELS)
        )
        self.nli_ensemble_majority   = int(nli_ensemble_majority)
        self._nli_ensemble: Optional[List[Tuple[str, Any]]] = None
        self.length_ratio_min        = float(length_ratio_min)
        self.length_ratio_max        = float(length_ratio_max)
        self.semantic_dedup_threshold = float(semantic_dedup_threshold)
        self.best_of_n               = max(1, int(best_of_n))
        self.best_of_n_temperature   = float(best_of_n_temperature)
        self.best_of_n_top_p         = float(best_of_n_top_p)
        self.max_per_family          = dict(MAX_PER_FAMILY)
        if max_per_family:
            self.max_per_family.update(max_per_family)

        # IMPORTANT (vLLM v1 fork/CUDA conflict): vLLM 0.8+ uses multiprocessing
        # `fork` by default; if CUDA is initialized in the main process BEFORE
        # the vLLM workers fork, every child crashes with
        # "Cannot re-initialize CUDA in forked subprocess".
        # SentenceTransformer touches CUDA on construction, so we MUST load
        # vLLM first and the SBERT scorer afterwards.
        if model_backend == "vllm":
            self._load_vllm(model_id or self.VLLM_MODEL_ID, quantization)
        else:
            self._load_llama()

        logger.info(f"Loading SBERT model: {sbert_model}")
        self.sbert = SentenceTransformer(sbert_model)
        logger.info("  SBERT loaded.")

    # ── Model loaders ─────────────────────────────────────────

    def _load_llama(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading LLaMA: {self.LLAMA_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LLAMA_MODEL_ID, use_fast=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.LLAMA_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()
        logger.info("  LLaMA loaded in bfloat16 with device_map='auto'.")

    def _load_vllm(self, model_id: str, quantization: Optional[str]):
        from transformers import AutoTokenizer
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise RuntimeError(
                "vLLM backend requested but vllm is not installed. "
                "pip install vllm"
            ) from e

        logger.info(
            f"Loading vLLM generator: {model_id} "
            f"(quant={quantization}, tp={self.tensor_parallel_size})"
        )
        self.model_id  = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs = dict(
            model=model_id,
            # tensor_parallel_size set via constructor; default 1 = single card.
            # On A30 × 2 with 70B AWQ, set 2 in the CLI / .env so vLLM shards
            # the model weights across both cards (each card holds ~17.5 GB).
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype="auto",
            trust_remote_code=True,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)
        self._SamplingParams = SamplingParams
        logger.info("  vLLM generator ready.")

    # ── Batched generation (vLLM) ─────────────────────────────

    def generate_batch(self, prompts: List[str]) -> List[str]:
        sp = self._SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, seed=42)
        outs = self.llm.generate(prompts, sp)
        return [self._clean_output(o.outputs[0].text) or "" for o in outs]

    def _build_chat_prompt(
        self, task: str, base_text: str, strategy_instruction: str, pool: str = "A"
    ) -> str:
        if task == "summarization":
            system_content = SUMM_SYSTEM_PROMPTS.get(pool, SUMM_SYSTEM_PROMPTS["A"])
        else:
            system_content = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{strategy_instruction}\n\nOriginal:\n{base_text}\n\nRephrased version:"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_single_vllm(
        self, task: str, base_text: str, strategy_instruction: str, pool: str = "A"
    ) -> Optional[str]:
        prompt = self._build_chat_prompt(task, base_text, strategy_instruction, pool=pool)
        return self.generate_batch([prompt])[0] or None

    # ── Single-candidate generation (llama) ──────────────────

    def _clean_output(self, text: str) -> str:
        text = re.sub(r'^(Rephrased\s+version\s*:\s*)', '', text, flags=re.IGNORECASE)
        text = text.strip().strip('"\'').strip()
        lines, clean = text.split("\n"), []
        for line in lines:
            s = line.strip()
            if not s:
                break
            if any(s.lower().startswith(p) for p in [
                "note:", "explanation:", "i ", "here ", "this ", "the original",
                "(note", "changes made", "key changes"
            ]):
                break
            clean.append(s)
        return " ".join(clean).strip()

    def _generate_single_llama(
        self, task: str, base_text: str, strategy_instruction: str, pool: str = "A"
    ) -> Optional[str]:
        if task == "summarization":
            system_content = SUMM_SYSTEM_PROMPTS.get(pool, SUMM_SYSTEM_PROMPTS["A"])
        else:
            system_content = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": (
                f"{strategy_instruction}\n\n"
                f"Original:\n{base_text}\n\n"
                f"Rephrased version:"
            )},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_len = input_ids.shape[1]
        input_ids = input_ids.to(next(self.model.parameters()).device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=300,
                temperature=0.85,
                top_p=0.92,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output[0][input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._clean_output(raw) or None

    def _generate_single(
        self, task: str, base_text: str, strategy_instruction: str, pool: str = "A"
    ) -> Optional[str]:
        if self.backend == "vllm":
            return self._generate_single_vllm(task, base_text, strategy_instruction, pool=pool)
        return self._generate_single_llama(task, base_text, strategy_instruction, pool=pool)

    # ── Summarization instruction pool ───────────────────────

    def generate_instruction_pool(
        self,
        n: int,
        canonical_instruction: str,
        logger: Optional[logging.Logger] = None,
    ) -> List[str]:
        """Generate n unique summarization instructions via freeform LLM call."""
        _log = logger or logging.getLogger(__name__)
        return self._instruction_pool_freeform(n, canonical_instruction, _log)

    def _instruction_pool_freeform(
        self, n: int, canonical: str, log: logging.Logger
    ) -> List[str]:
        from sentence_transformers import util as _st_util

        prompt_text = (
            f"Generate {n} distinct instructions for asking someone to summarize a news "
            "article in 3-4 sentences, capturing the main events and key details. "
            "Vary the wording, tone, and sentence structure (formal, casual, imperative, "
            "question-form, passive, etc.). "
            "Output ONLY the instructions, one per line, with no numbering, bullets, or extra text."
        )
        messages = [
            {"role": "system", "content": "You generate diverse rewordings of task instructions."},
            {"role": "user",   "content": prompt_text},
        ]

        if self.backend == "vllm":
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sp = self._SamplingParams(temperature=0.9, top_p=0.95, max_tokens=n * 25, seed=42)
            raw = self.llm.generate([full_prompt], sp)[0].outputs[0].text
        else:  # llama
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            input_len = input_ids.shape[1]
            device = next(self.model.parameters()).device
            with torch.no_grad():
                out = self.model.generate(
                    input_ids.to(device),
                    max_new_tokens=n * 25,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            raw = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        candidates: List[str] = []
        for line in raw.split("\n"):
            line = re.sub(r'^\s*\d+[.)]\s*', '', line)
            line = re.sub(r'^\s*[-*•]\s*', '', line).strip()
            if len(line.split()) >= 5:
                candidates.append(line)

        pool: List[str] = [canonical]
        seen: set = {self._first_n_words_sig(canonical)}
        canon_emb = self.sbert.encode(canonical, convert_to_tensor=True)
        for cand in candidates:
            if len(pool) >= n:
                break
            sig = self._first_n_words_sig(cand)
            if sig in seen:
                continue
            emb = self.sbert.encode(cand, convert_to_tensor=True)
            sim = float(_st_util.cos_sim(canon_emb, emb).item())
            if self.similarity_threshold <= sim <= self.similarity_upper:
                if self._token_overlap(cand, canonical) <= self.max_token_overlap:
                    pool.append(cand)
                    seen.add(sig)

        log.info(
            f"  Freeform instruction pool: {len(pool)} passed SBERT filter "
            f"({len(candidates)} lines parsed from LLM output)"
        )

        if len(pool) < n:
            log.warning(
                f"  Freeform pool too small ({len(pool)} < {n}); "
                "supplementing with strategy-based generation."
            )
            supplement = self._instruction_pool_strategies(n - len(pool), canonical, log)
            for s in supplement:
                if len(pool) >= n:
                    break
                sig = self._first_n_words_sig(s)
                if sig not in seen:
                    pool.append(s)
                    seen.add(sig)

        return pool

    def _instruction_pool_strategies(
        self, n: int, canonical: str, log: logging.Logger
    ) -> List[str]:
        dummy = {
            "instance_id": "_summ_instr_pool",
            "task":        "summarization",
            "base_text":   canonical,
            "base_prompt": canonical,
            "metadata":    {},
        }
        orig_max = self.max_per_strategy
        self.max_per_strategy = max(orig_max, (n // len(STRATEGY_INSTRUCTIONS)) + 2)
        try:
            paraphrases = self.generate_paraphrases(dummy, n_variants=n - 1, max_retries=5)
        finally:
            self.max_per_strategy = orig_max
        pool = [canonical] + [p["paraphrased_text"] for p in paraphrases]
        log.info(f"  Strategy instruction pool: {len(pool)} generated (target={n})")
        return pool

    # ── SBERT helpers ─────────────────────────────────────────

    def _compute_similarity(self, original: str, candidate: str) -> float:
        embeddings = self.sbert.encode([original, candidate], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())

    @staticmethod
    def _first_n_words_sig(text: str, n: int = 8) -> str:
        return " ".join(text.lower().split()[:n])

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # ── New Tier-1 filter helpers (PUBLICATION_SPEC §6.1–6.3) ─────────────

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\w+", text))

    def _length_ratio(self, candidate: str, base: str) -> float:
        """word_count(candidate) / word_count(base). 1.0 = same length."""
        b = self._word_count(base)
        if b == 0:
            return 0.0
        return self._word_count(candidate) / b

    def _passes_length_ratio(self, candidate: str, base: str) -> Tuple[bool, float]:
        """§6.2 — accept only when 0.7× ≤ |cand|/|base| ≤ 1.5×."""
        r = self._length_ratio(candidate, base)
        return (self.length_ratio_min <= r <= self.length_ratio_max), r

    def _passes_semantic_dedup(
        self,
        candidate: str,
        accepted_variants: List[Dict[str, Any]],
    ) -> Tuple[bool, float]:
        """§6.3 — reject if SBERT cos(candidate, any accepted) > threshold.

        Returns (passes, max_cos_to_accepted). Empty accepted-set passes
        trivially with max_cos = 0.0.
        """
        if not accepted_variants:
            return True, 0.0
        cand_emb = self.sbert.encode(candidate, convert_to_tensor=True)
        max_cos = 0.0
        for v in accepted_variants:
            acc_emb = self.sbert.encode(v["paraphrased_text"], convert_to_tensor=True)
            cos = float(util.cos_sim(cand_emb, acc_emb).item())
            if cos > max_cos:
                max_cos = cos
            if cos > self.semantic_dedup_threshold:
                return False, cos
        return True, max_cos

    def _bidirectional_nli(
        self,
        base: str,
        candidate: str,
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[Dict[str, Any]]]:
        """§6.1 + §7.1 — bidirectional NLI entailment check.

        Returns ``(passes, P_fwd, P_bwd, ensemble_audit)`` where:
          - ``passes``        — overall accept/reject decision
          - ``P_fwd / P_bwd`` — entailment probability from the primary
                                model (single-model mode) or the
                                ensemble mean (ensemble mode)
          - ``ensemble_audit`` — dict with per-model probs and vote
                                  counts when the ensemble is active;
                                  None otherwise

        Modes:
          (a) gate disabled → ``(True, None, None, None)``
          (b) single-model  → primary NLI, as in §6.1
          (c) ensemble      → §7.1 — vote ≥ `majority` in BOTH directions

        Graceful degradation: any single model load failure becomes an
        abstention (does not count toward the majority). All-fail returns
        ``(True, None, None, None)`` so we never block generation on
        infrastructure failures — the per-variant audit metadata flags
        the bypass so reviewers can see it.
        """
        if not self.enable_nli_gate:
            return True, None, None, None

        # Ensemble mode (§7.1)
        if self.enable_nli_ensemble:
            if self._nli_ensemble is None:
                self._nli_ensemble = _get_nli_ensemble(self.nli_ensemble_models)
            fwd = _nli_ensemble_entail(
                self._nli_ensemble, base, candidate,
                self.nli_entail_threshold, self.nli_ensemble_majority,
            )
            bwd = _nli_ensemble_entail(
                self._nli_ensemble, candidate, base,
                self.nli_entail_threshold, self.nli_ensemble_majority,
            )
            # All-abstain → bypass (graceful degradation).
            if fwd["n_models"] == 0 or bwd["n_models"] == 0:
                return True, None, None, None
            passes = fwd["passes"] and bwd["passes"]
            # Compute the mean-of-loaded-models entailment prob for legacy
            # consumers that still expect a scalar.
            def _mean_prob(audit):
                vals = [v for v in audit["per_model"].values() if v is not None]
                return sum(vals) / len(vals) if vals else None
            p_fwd_mean = _mean_prob(fwd)
            p_bwd_mean = _mean_prob(bwd)
            return passes, p_fwd_mean, p_bwd_mean, {
                "ensemble_majority": self.nli_ensemble_majority,
                "ensemble_size":     fwd["n_models"],
                "fwd_votes":         fwd["votes"],
                "bwd_votes":         bwd["votes"],
                "fwd_per_model":     fwd["per_model"],
                "bwd_per_model":     bwd["per_model"],
            }

        # Single-model mode (§6.1)
        if self._nli_model is None:
            self._nli_model = _get_nli_model(self.nli_model_name)
        if self._nli_model is None:
            return True, None, None, None  # graceful degradation
        p_fwd = _nli_entail_prob(self._nli_model, base, candidate)
        p_bwd = _nli_entail_prob(self._nli_model, candidate, base)
        if p_fwd is None or p_bwd is None:
            return True, p_fwd, p_bwd, None  # graceful degradation
        passes = (
            p_fwd >= self.nli_entail_threshold
            and p_bwd >= self.nli_entail_threshold
        )
        return passes, p_fwd, p_bwd, None

    # ── Best-of-N per strategy (PUBLICATION_SPEC §6.5) ─────────────────────

    def _generate_n_vllm(
        self,
        task: str,
        base_text: str,
        strategy_instruction: str,
        n: int,
        seed: int,
        pool: str = "A",
    ) -> List[str]:
        """Generate n candidates from vLLM in one call with diverse sampling.

        Each call uses `temperature=best_of_n_temperature`,
        `top_p=best_of_n_top_p`, and per-call seed derived from the strategy
        index so paraphrase sets are reproducible (seed=42 → seed=42+s_idx).
        Returns the list of cleaned candidate strings.
        """
        if self.backend != "vllm" or n <= 1:
            single = self._generate_single(task, base_text, strategy_instruction, pool=pool)
            return [single] if single else []
        prompt = self._build_chat_prompt(task, base_text, strategy_instruction, pool=pool)
        sp = self._SamplingParams(
            temperature=self.best_of_n_temperature,
            top_p=self.best_of_n_top_p,
            max_tokens=256,
            n=n,
            seed=seed,
        )
        try:
            outs = self.llm.generate([prompt], sp)
            cands = []
            for o in outs[0].outputs:
                txt = self._clean_output(o.text)
                if txt:
                    cands.append(txt)
            return cands
        except Exception as e:  # noqa: BLE001
            # vLLM versions that don't support `n` in SamplingParams will
            # error here — fall back to n=1.
            logger.debug(f"vLLM best-of-N failed ({e}); falling back to n=1")
            single = self._generate_single(task, base_text, strategy_instruction, pool=pool)
            return [single] if single else []

    def _select_most_diverse(
        self,
        candidates: List[Dict[str, Any]],
        accepted_variants: List[Dict[str, Any]],
        base_text: str,
    ) -> Dict[str, Any]:
        """Pick the candidate with lowest token Jaccard to base + accepted set.

        candidates: list of pre-filtered candidate metadata dicts
        accepted_variants: already-accepted variants from this instance
        base_text: the original prompt
        """
        if len(candidates) == 1:
            return candidates[0]
        # Score = mean Jaccard overlap to (base + every accepted variant).
        # Lower score = more diverse.
        ref_texts = [base_text] + [v["paraphrased_text"] for v in accepted_variants]
        def diversity_score(c):
            text = c["paraphrased_text"]
            overlaps = [self._token_overlap(text, r) for r in ref_texts]
            return sum(overlaps) / len(overlaps)
        return min(candidates, key=diversity_score)

    # ── Core: generate paraphrases ────────────────────────────

    def generate_paraphrases(
        self,
        record:      Dict[str, Any],
        n_variants:  int = 8,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate up to n_variants filtered paraphrases for one record.

        Pipeline (PUBLICATION_SPEC §6):
          generate (best-of-N) → length-band → SBERT band → token-overlap
            → semantic dedup → bidirectional NLI → family-budget cap → accept

        Each accepted variant carries the full audit-metadata dict (NLI fwd/bwd,
        SBERT cos, length ratio, token overlap, strategy family, best-of-N idx).
        """
        base_text   = record["base_text"]
        task        = record["task"]
        instance_id = record["instance_id"]
        pool        = record.get("pool", "A")  # "A" = open-style, "B" = constraint-bearing

        valid_paraphrases: List[Dict[str, Any]] = []
        seen_sigs: set = {self._first_n_words_sig(base_text)}
        # §6.6 — cap by family, not by strategy.
        family_counts: Dict[str, int] = {}
        # Per-instance reject diagnostics (logged on completion).
        reject_counts: Dict[str, int] = {
            "empty_or_short": 0, "identical_to_base": 0, "duplicate_sig": 0,
            "out_of_sbert_band": 0, "too_high_token_overlap": 0,
            "length_ratio_oob": 0, "semantic_dedup_collision": 0,
            "nli_failed": 0, "family_budget_exceeded": 0,
        }

        allowed_strategies = STRATEGY_BY_TASK.get(task, STRATEGY_NAMES)
        # Preserve the original (name, instruction, idx-in-master-list) tuple so
        # best-of-N seeding is deterministic per strategy.
        task_strategy_pairs = [
            (idx, name, instr)
            for idx, (name, instr) in enumerate(zip(STRATEGY_NAMES, STRATEGY_INSTRUCTIONS))
            if name in allowed_strategies
        ]

        effective_threshold = self.task_thresholds.get(task, self.similarity_threshold)

        for retry in range(max_retries):
            if len(valid_paraphrases) >= n_variants:
                break

            if retry > 0:
                logger.info(
                    f"  [{instance_id}] Retry {retry}: "
                    f"{len(valid_paraphrases)}/{n_variants} — re-trying allowed strategies"
                )

            for strategy_idx, strategy_name, strategy_instruction in task_strategy_pairs:
                if len(valid_paraphrases) >= n_variants:
                    break

                family = STRATEGY_FAMILY.get(strategy_name, "lexical")
                if family_counts.get(family, 0) >= self.max_per_family.get(family, 999):
                    reject_counts["family_budget_exceeded"] += 1
                    continue
                # FORMAT axis is near-copy by design → skip the diversity gates
                # (upper SBERT bound, token-overlap, semantic/signature dedup).
                # It must still clear the lower SBERT bound + bidirectional NLI.
                diversity_exempt = family in DIVERSITY_EXEMPT_FAMILIES

                try:
                    # §6.5 — best-of-N generation: ask for n candidates,
                    # filter all of them, pick the most diverse survivor.
                    seed = 42 + strategy_idx + retry * 100
                    candidates_raw = self._generate_n_vllm(
                        task, base_text, strategy_instruction,
                        n=self.best_of_n, seed=seed, pool=pool,
                    )

                    qualified: List[Dict[str, Any]] = []
                    for bon_idx, candidate in enumerate(candidates_raw):
                        if not candidate or len(candidate.split()) < self.min_word_count:
                            reject_counts["empty_or_short"] += 1
                            continue
                        if candidate.strip().lower() == base_text.strip().lower():
                            reject_counts["identical_to_base"] += 1
                            continue
                        sig = self._first_n_words_sig(candidate)
                        if not diversity_exempt and sig in seen_sigs:
                            reject_counts["duplicate_sig"] += 1
                            continue

                        # §6.2 — length-ratio filter
                        len_ok, len_ratio = self._passes_length_ratio(candidate, base_text)
                        if not len_ok:
                            reject_counts["length_ratio_oob"] += 1
                            continue

                        # SBERT band. The LOWER bound (meaning preserved) always
                        # applies; the UPPER ceiling is a diversity gate, skipped
                        # for the near-copy FORMAT axis.
                        similarity = self._compute_similarity(base_text, candidate)
                        if similarity < effective_threshold:
                            reject_counts["out_of_sbert_band"] += 1
                            continue
                        if not diversity_exempt and similarity > self.similarity_upper:
                            reject_counts["out_of_sbert_band"] += 1
                            continue

                        # Token Jaccard floor vs base + accepted variants
                        # (diversity gate → skipped for the FORMAT axis).
                        tov_base = self._token_overlap(candidate, base_text)
                        if not diversity_exempt:
                            if tov_base > self.max_token_overlap:
                                reject_counts["too_high_token_overlap"] += 1
                                continue
                            if any(
                                self._token_overlap(candidate, v["paraphrased_text"]) > self.max_token_overlap
                                for v in valid_paraphrases
                            ):
                                reject_counts["too_high_token_overlap"] += 1
                                continue

                        # §6.3 — SBERT semantic dedup vs accepted variants
                        # (diversity gate → skipped for the FORMAT axis).
                        if diversity_exempt:
                            max_cos = 0.0
                        else:
                            dedup_ok, max_cos = self._passes_semantic_dedup(
                                candidate, valid_paraphrases
                            )
                            if not dedup_ok:
                                reject_counts["semantic_dedup_collision"] += 1
                                continue

                        # §6.1 + §7.1 — bidirectional NLI gate on ALL tasks
                        nli_ok, p_fwd, p_bwd, nli_ensemble_audit = (
                            self._bidirectional_nli(base_text, candidate)
                        )
                        if not nli_ok:
                            reject_counts["nli_failed"] += 1
                            logger.debug(
                                f"  [{instance_id}] NLI rejected '{strategy_name}' "
                                f"(fwd={p_fwd}, bwd={p_bwd})"
                            )
                            continue

                        variant_record = {
                            "variant_idx":           len(valid_paraphrases),
                            "paraphrased_text":      candidate,
                            "strategy":              strategy_name,
                            "strategy_family":       family,
                            "sbert_similarity":      round(similarity, 4),
                            "token_overlap_to_base": round(tov_base, 4),
                            "length_ratio":          round(len_ratio, 4),
                            "max_cos_to_accepted":   round(max_cos, 4),
                            "nli_entail_fwd":        round(p_fwd, 4) if p_fwd is not None else None,
                            "nli_entail_bwd":        round(p_bwd, 4) if p_bwd is not None else None,
                            "nli_passed":            (p_fwd is not None and p_bwd is not None),
                            "best_of_n_index":       bon_idx,
                        }
                        # §7.1 — when the ensemble is active, persist the
                        # full audit (per-model probs + vote counts) so
                        # reviewers can verify the majority decision.
                        if nli_ensemble_audit is not None:
                            variant_record["nli_ensemble"] = nli_ensemble_audit
                        qualified.append(variant_record)

                    if not qualified:
                        continue

                    # §6.5 — select most diverse among qualifying candidates
                    chosen = self._select_most_diverse(qualified, valid_paraphrases, base_text)
                    chosen["variant_idx"] = len(valid_paraphrases)
                    seen_sigs.add(self._first_n_words_sig(chosen["paraphrased_text"]))
                    family_counts[family] = family_counts.get(family, 0) + 1
                    valid_paraphrases.append(chosen)

                except Exception as e:
                    logger.warning(f"  [{instance_id}] Strategy '{strategy_name}' failed: {e}")

        # Diagnostic logging — per-instance reject breakdown helps reviewers
        # (and us) understand which gates dominate the rejection budget.
        n_reject = sum(reject_counts.values())
        if n_reject > 0:
            logger.info(
                f"  [{instance_id}] rejection breakdown ({n_reject} total): "
                + ", ".join(f"{k}={v}" for k, v in reject_counts.items() if v > 0)
            )

        for i, v in enumerate(valid_paraphrases):
            v["variant_idx"] = i

        return valid_paraphrases[:n_variants]

    # ── Build full prompt variants ────────────────────────────

    def build_prompt_variants(
        self,
        record:      Dict[str, Any],
        paraphrases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        base_text   = record["base_text"]
        base_prompt = record["base_prompt"]

        for variant in paraphrases:
            pt = variant["paraphrased_text"]
            if base_text in base_prompt:
                variant["full_prompt"] = base_prompt.replace(base_text, pt, 1)
            else:
                variant["full_prompt"] = pt + "\n\n" + base_prompt

        return paraphrases

    # ── Combined pipeline for one instance ───────────────────

    def process_instance(
        self,
        record:     Dict[str, Any],
        n_variants: int = 8,
    ) -> Dict[str, Any]:
        t_start = time.time()

        paraphrases = self.generate_paraphrases(record, n_variants=n_variants)
        paraphrases = self.build_prompt_variants(record, paraphrases)

        elapsed  = time.time() - t_start
        n_gen    = len(paraphrases)
        mean_sim = (
            float(np.mean([p["sbert_similarity"] for p in paraphrases]))
            if paraphrases else 0.0
        )
        diversity = self._diversity_stats([p["paraphrased_text"] for p in paraphrases])
        n_strat   = len({p["strategy"] for p in paraphrases})
        diversity["n_strategies"] = n_strat

        logger.info(
            f"  [{record['instance_id']}] n_variants={n_gen}, "
            f"mean_sim={mean_sim:.3f}, distinct2={diversity['distinct_2']:.3f}, "
            f"mean_overlap={diversity['mean_pairwise_overlap']:.3f}, "
            f"n_strategies={n_strat}, time={elapsed:.1f}s"
        )

        return {
            "instance_id":          record["instance_id"],
            "task":                 record["task"],
            "base_text":            record["base_text"],
            "base_prompt":          record["base_prompt"],
            "metadata":             record["metadata"],
            "variants":             paraphrases,
            "n_variants_generated": n_gen,
            "generation_time_s":    round(elapsed, 2),
            "diversity":            diversity,
        }

    def _diversity_stats(self, texts: List[str]) -> Dict[str, Any]:
        if len(texts) < 2:
            return {"distinct_2": 0.0, "mean_pairwise_overlap": 0.0, "n_strategies": 0}
        bigrams, total = set(), 0
        for t in texts:
            toks = re.findall(r"\w+", t.lower())
            for i in range(len(toks) - 1):
                bigrams.add((toks[i], toks[i + 1]))
                total += 1
        distinct_2 = len(bigrams) / total if total else 0.0
        overlaps = [
            self._token_overlap(texts[i], texts[j])
            for i in range(len(texts)) for j in range(i + 1, len(texts))
        ]
        return {
            "distinct_2": round(distinct_2, 4),
            "mean_pairwise_overlap": round(float(np.mean(overlaps)), 4),
            "n_strategies": 0,
        }
