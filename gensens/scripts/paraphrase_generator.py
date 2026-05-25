"""
GenSens Paraphrase Generator
=============================
Supports two generation backends:

  "local"  (default) → google/flan-t5-large      (~770 MB, CPU/MPS, for development on Mac)
  "llama"            → meta-llama/Meta-Llama-3.1-8B-Instruct  (bfloat16, H100, for production)

Both backends share:
  • 16 diverse strategy instructions
  • Task-specific system prompts
  • SBERT "all-mpnet-base-v2" similarity filter ≥ 0.82
  • Deduplication by first-8-word signature
  • Up to 3 retries if < n_variants valid paraphrases found
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Task-specific system prompts  (used by LLaMA chat template)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "summarization": (
        "You are a prompt rewriting assistant. Rephrase summarization instructions "
        "while keeping the exact same meaning and intent. Do NOT change what is being "
        "asked — only change the wording. Return ONLY the rephrased instruction, nothing else."
    ),
    "code": (
        "You are a problem statement rewriting assistant. Rephrase coding problem "
        "descriptions while keeping the exact same functional requirements. The rephrased "
        "version must ask for the same function behavior. Return ONLY the rephrased problem "
        "description, nothing else."
    ),
    "creative": (
        "You are a creative writing prompt rewriter. Rephrase story prompts while keeping "
        "the same core scenario and creative direction. Do NOT add major new story elements. "
        "Return ONLY the rephrased story prompt, nothing else."
    ),
    "dialogue": (
        "You are a dialogue rewriting assistant. Rephrase user questions while keeping the "
        "exact same intent and information being requested. Return ONLY the rephrased "
        "question, nothing else."
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
]

STRATEGY_NAMES: List[str] = [
    "formal_tone", "casual_tone", "reordered_clauses", "synonyms",
    "concise", "different_structure", "different_opening", "role_prefix",
    "passive_voice", "question_form", "imperative", "elaborated",
    "technical_vocab", "simple_vocab", "split_sentences", "merged_sentences",
]

assert len(STRATEGY_INSTRUCTIONS) == len(STRATEGY_NAMES) == 16


# ─────────────────────────────────────────────────────────────
# Flan-T5 prompt builder  (plain text, no chat template)
# ─────────────────────────────────────────────────────────────

def _build_flan_prompt(task: str, base_text: str, strategy_instruction: str) -> str:
    task_context = {
        "summarization": "a summarization instruction",
        "code":          "a coding problem description",
        "creative":      "a creative writing prompt",
        "dialogue":      "a user dialogue question",
    }.get(task, "a text")

    return (
        f"{strategy_instruction}\n\n"
        f"This is {task_context}.\n\n"
        f"Original: {base_text}\n\n"
        f"Rephrased version:"
    )


# ─────────────────────────────────────────────────────────────
# ParaphraseGenerator
# ─────────────────────────────────────────────────────────────

class ParaphraseGenerator:
    """
    Generates paraphrase variants using LLaMA-3.1-8B-Instruct (H100)
    or Flan-T5-large (local Mac/CPU), filtered with SBERT cos-sim ≥ 0.82.

    Args:
        model_backend : "local"  → google/flan-t5-large   (CPU/MPS friendly)
                        "llama"  → Meta-Llama-3.1-8B-Instruct (H100, bfloat16)
        sbert_model   : SBERT model name (default: all-mpnet-base-v2)
        similarity_threshold : minimum SBERT cos-sim to keep a candidate (default: 0.82)
        min_word_count       : minimum words in a valid paraphrase (default: 5)
    """

    LOCAL_MODEL_ID = "google/flan-t5-large"
    LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __init__(
        self,
        model_backend:        str   = "local",
        sbert_model:          str   = "all-mpnet-base-v2",
        similarity_threshold: float = 0.82,
        min_word_count:       int   = 5,
    ):
        assert model_backend in ("local", "llama"), \
            "model_backend must be 'local' or 'llama'"
        self.backend              = model_backend
        self.similarity_threshold = similarity_threshold
        self.min_word_count       = min_word_count

        # Load SBERT first (small, always needed)
        logger.info(f"Loading SBERT model: {sbert_model}")
        self.sbert = SentenceTransformer(sbert_model)
        logger.info("  SBERT loaded.")

        # Load generation model
        if model_backend == "local":
            self._load_flan()
        else:
            self._load_llama()

    # ── Model loaders ─────────────────────────────────────────

    def _load_flan(self):
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        logger.info(f"Loading Flan-T5: {self.LOCAL_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LOCAL_MODEL_ID, use_fast=True
        )
        # Prefer MPS on Apple Silicon; fall back to CPU
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.LOCAL_MODEL_ID
        ).to(self._device)
        self.model.eval()
        logger.info(f"  Flan-T5 loaded on {self._device}.")

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
            torch_dtype  = torch.bfloat16,
            device_map   = "auto",
            local_files_only = True,
        )
        self.model.eval()
        logger.info("  LLaMA loaded in bfloat16 with device_map='auto'.")

    # ── Single-candidate generation ───────────────────────────

    def _clean_output(self, text: str) -> str:
        """Strip echoed prefixes, quotes, and model self-commentary."""
        text = re.sub(r'^(Rephrased\s+version\s*:\s*)', '', text, flags=re.IGNORECASE)
        text = text.strip().strip('"\'').strip()
        # Stop at blank line or model self-commentary
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

    def _generate_single_flan(
        self, task: str, base_text: str, strategy_instruction: str
    ) -> Optional[str]:
        prompt = _build_flan_prompt(task, base_text, strategy_instruction)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens = 200,
                num_beams      = 4,
                early_stopping = True,
                temperature    = 0.85,
                do_sample      = True,
                top_p          = 0.92,
            )
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._clean_output(raw) or None

    def _generate_single_llama(
        self, task: str, base_text: str, strategy_instruction: str
    ) -> Optional[str]:
        messages = [
            {
                "role":    "system",
                "content": SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"]),
            },
            {
                "role":    "user",
                "content": (
                    f"{strategy_instruction}\n\n"
                    f"Original:\n{base_text}\n\n"
                    f"Rephrased version:"
                ),
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_len = input_ids.shape[1]
        input_ids = input_ids.to(next(self.model.parameters()).device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens = 300,
                temperature    = 0.85,
                top_p          = 0.92,
                do_sample      = True,
                pad_token_id   = self.tokenizer.pad_token_id,
            )

        new_tokens = output[0][input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._clean_output(raw) or None

    def _generate_single(
        self, task: str, base_text: str, strategy_instruction: str
    ) -> Optional[str]:
        if self.backend == "local":
            return self._generate_single_flan(task, base_text, strategy_instruction)
        return self._generate_single_llama(task, base_text, strategy_instruction)

    # ── SBERT helpers ─────────────────────────────────────────

    def _compute_similarity(self, original: str, candidate: str) -> float:
        embeddings = self.sbert.encode(
            [original, candidate], convert_to_tensor=True
        )
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())

    @staticmethod
    def _first_n_words_sig(text: str, n: int = 8) -> str:
        return " ".join(text.lower().split()[:n])

    # ── Core: generate paraphrases ────────────────────────────

    def generate_paraphrases(
        self,
        record:      Dict[str, Any],
        n_variants:  int = 8,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate up to n_variants filtered paraphrases for one record.
        Uses all 16 strategies; retries up to max_retries if not enough pass.
        """
        base_text   = record["base_text"]
        task        = record["task"]
        instance_id = record["instance_id"]

        valid_paraphrases: List[Dict[str, Any]] = []
        seen_sigs: set = {self._first_n_words_sig(base_text)}

        for retry in range(max_retries):
            if len(valid_paraphrases) >= n_variants:
                break

            if retry > 0:
                logger.info(
                    f"  [{instance_id}] Retry {retry}: "
                    f"{len(valid_paraphrases)}/{n_variants} — trying all strategies again"
                )

            for strat_idx, strategy_instruction in enumerate(STRATEGY_INSTRUCTIONS):
                if len(valid_paraphrases) >= n_variants:
                    break

                strategy_name = STRATEGY_NAMES[strat_idx]
                try:
                    candidate = self._generate_single(
                        task, base_text, strategy_instruction
                    )
                    if not candidate:
                        continue

                    wc = len(candidate.split())
                    if wc < self.min_word_count:
                        continue

                    if candidate.strip().lower() == base_text.strip().lower():
                        continue   # exact copy

                    sig = self._first_n_words_sig(candidate)
                    if sig in seen_sigs:
                        continue   # near-duplicate

                    similarity = self._compute_similarity(base_text, candidate)
                    if similarity < self.similarity_threshold:
                        logger.debug(
                            f"  [{instance_id}] '{strategy_name}' sim={similarity:.3f} — rejected"
                        )
                        continue

                    seen_sigs.add(sig)
                    valid_paraphrases.append({
                        "variant_idx":      len(valid_paraphrases),
                        "paraphrased_text": candidate,
                        "sbert_similarity": round(similarity, 4),
                        "strategy":         strategy_name,
                    })

                except Exception as e:
                    logger.warning(
                        f"  [{instance_id}] Strategy '{strategy_name}' failed: {e}"
                    )

        # Re-index
        for i, v in enumerate(valid_paraphrases):
            v["variant_idx"] = i

        return valid_paraphrases[:n_variants]

    # ── Build full prompt variants ────────────────────────────

    def build_prompt_variants(
        self,
        record:       Dict[str, Any],
        paraphrases:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Replace base_text in base_prompt with each paraphrased_text
        to build the full paraphrased prompts.
        """
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
        """
        Full pipeline for a single instance:
        generate → filter → build full prompts → return enriched record.
        """
        t_start = time.time()

        paraphrases = self.generate_paraphrases(record, n_variants=n_variants)
        paraphrases = self.build_prompt_variants(record, paraphrases)

        elapsed    = time.time() - t_start
        n_gen      = len(paraphrases)
        mean_sim   = (
            float(np.mean([p["sbert_similarity"] for p in paraphrases]))
            if paraphrases else 0.0
        )

        logger.info(
            f"  [{record['instance_id']}] n_variants={n_gen}, "
            f"mean_sim={mean_sim:.3f}, time={elapsed:.1f}s"
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
        }
