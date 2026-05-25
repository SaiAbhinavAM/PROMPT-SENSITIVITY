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
    # Default H100 generator (override via model_id). 70B must be a quantized
    # (AWQ/GPTQ-INT4) repo to fit a single 80GB card; bf16 70B needs >1 GPU.
    VLLM_MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

    def __init__(
        self,
        model_backend:        str   = "local",
        sbert_model:          str   = "all-mpnet-base-v2",
        similarity_threshold: float = 0.82,
        min_word_count:       int   = 5,
        # --- Diversity controls (quality fix) ---
        similarity_upper:     float = 0.98,   # reject near-identical restatements
        max_token_overlap:    float = 0.85,   # reject lexically-too-close candidates
        max_per_strategy:     int   = 2,      # cap variants from any single strategy
        # --- vLLM backend (H100) ---
        model_id:             Optional[str] = None,
        quantization:         Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len:        int   = 4096,
    ):
        assert model_backend in ("local", "llama", "vllm"), \
            "model_backend must be 'local', 'llama', or 'vllm'"
        self.backend              = model_backend
        self.similarity_threshold = similarity_threshold
        self.min_word_count       = min_word_count
        self.similarity_upper     = similarity_upper
        self.max_token_overlap    = max_token_overlap
        self.max_per_strategy     = max_per_strategy
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len        = max_model_len

        # Load SBERT first (small, always needed)
        logger.info(f"Loading SBERT model: {sbert_model}")
        self.sbert = SentenceTransformer(sbert_model)
        logger.info("  SBERT loaded.")

        # Load generation model
        if model_backend == "local":
            self._load_flan()
        elif model_backend == "vllm":
            self._load_vllm(model_id or self.VLLM_MODEL_ID, quantization)
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

    def _load_vllm(self, model_id: str, quantization: Optional[str]):
        """Load a generator via vLLM (H100). Batched, 10-50x faster than HF.

        For a 70B target on one 80GB card, pass an AWQ/GPTQ-INT4 repo and
        quantization="awq_marlin" (or "gptq_marlin"); ~40GB weights leave room
        for the KV cache. vLLM is imported lazily so the laptop flan-t5 path
        does not require vllm to be installed.
        """
        from transformers import AutoTokenizer
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover - H100-only path
            raise RuntimeError(
                "vLLM backend requested but vllm is not installed. "
                "On the H100: pip install vllm"
            ) from e

        logger.info(f"Loading vLLM generator: {model_id} (quant={quantization})")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs = dict(
            model=model_id,
            tensor_parallel_size=1,           # single 80GB H100
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

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Batched generation for the vLLM backend (one engine call)."""
        sp = self._SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=256, seed=42,
        )
        outs = self.llm.generate(prompts, sp)
        return [self._clean_output(o.outputs[0].text) or "" for o in outs]

    def _build_chat_prompt(self, task: str, base_text: str, strategy_instruction: str) -> str:
        """Render the instruct chat template to a single string for vLLM."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["summarization"])},
            {"role": "user", "content": f"{strategy_instruction}\n\nOriginal:\n{base_text}\n\nRephrased version:"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_single_vllm(self, task: str, base_text: str, strategy_instruction: str) -> Optional[str]:
        prompt = self._build_chat_prompt(task, base_text, strategy_instruction)
        return self.generate_batch([prompt])[0] or None

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
        if self.backend == "vllm":
            return self._generate_single_vllm(task, base_text, strategy_instruction)
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

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        """Jaccard overlap of lowercased word sets — a lexical-similarity proxy.

        High overlap means the candidate reuses the same words (low lexical
        diversity) even if SBERT cosine is acceptable.
        """
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

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
        strategy_counts: Dict[str, int] = {}

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

                # Diversity: cap how many variants any single strategy contributes.
                if strategy_counts.get(strategy_name, 0) >= self.max_per_strategy:
                    continue
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
                    # Similarity BAND: too low → meaning drift; too high → trivial
                    # restatement that adds no real lexical perturbation.
                    if similarity < self.similarity_threshold or similarity > self.similarity_upper:
                        logger.debug(
                            f"  [{instance_id}] '{strategy_name}' sim={similarity:.3f} — out of band"
                        )
                        continue

                    # Lexical-divergence floor: reject candidates that reuse too
                    # many of the base's words, or are near-lexical-dupes of an
                    # already-accepted variant.
                    if self._token_overlap(candidate, base_text) > self.max_token_overlap:
                        continue
                    if any(
                        self._token_overlap(candidate, v["paraphrased_text"]) > self.max_token_overlap
                        for v in valid_paraphrases
                    ):
                        continue

                    seen_sigs.add(sig)
                    strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
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
        diversity = self._diversity_stats([p["paraphrased_text"] for p in paraphrases])
        n_strat = len({p["strategy"] for p in paraphrases})
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
        """Corpus-level lexical diversity over a set of variants.

        distinct_2: fraction of unique bigrams (higher = more diverse).
        mean_pairwise_overlap: mean token-Jaccard between variant pairs
                               (lower = more diverse).
        """
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
            "n_strategies": 0,  # filled by caller
        }
