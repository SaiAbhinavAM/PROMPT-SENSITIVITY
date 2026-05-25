import os
import time
import torch
from typing import List
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import logging

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

# Architecture families that use encoder-decoder (Seq2Seq)
_SEQ2SEQ_ARCHS = {
    "t5", "bart", "mbart", "pegasus", "marian", "longt5",
    "led", "prophetnet", "mt5", "nllb", "flan",
}


def _is_seq2seq(model_name: str) -> bool:
    """Return True if the model uses an encoder-decoder architecture."""
    name_lower = model_name.lower()
    if any(k in name_lower for k in _SEQ2SEQ_ARCHS):
        return True
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        arch = getattr(cfg, "model_type", "").lower()
        return any(k in arch for k in _SEQ2SEQ_ARCHS)
    except Exception:
        return False


class ModelInterface:
    def __init__(self, model_name: str, config: Config):
        self.config = config
        self.model_name = model_name
        self.is_seq2seq = _is_seq2seq(model_name)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"\n--- Running model: {model_name} ({'seq2seq' if self.is_seq2seq else 'causal'}) on {self.device} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_responses(self, prompts: List[str]) -> List[str]:
        start_t = time.time()

        formatted_prompts = []
        for p in prompts:
            if "flan-t5" in self.model_name.lower() and not p.lower().startswith("summarize:"):
                formatted_prompts.append(f"summarize: {p}")
            else:
                formatted_prompts.append(p)

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.is_seq2seq:
            # Seq2Seq: output tokens are purely the generated sequence
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            # Causal LM: output contains the prompt tokens; strip them
            prompt_lengths = inputs["input_ids"].shape[1]
            responses = self.tokenizer.batch_decode(
                outputs[:, prompt_lengths:], skip_special_tokens=True
            )

        print(f"[{self.model_name}] Generated {len(prompts)} responses in {time.time() - start_t:.2f}s")
        return [r.strip() for r in responses]

    def _compute_ppl(self, prompt: str, response: str) -> float:
        """Compute per-sample perplexity for one (prompt, response) pair."""
        import math

        if self.is_seq2seq:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            dec = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            if enc["input_ids"].shape[1] == 0 or dec["input_ids"].shape[1] == 0:
                return None
            with torch.no_grad():
                out = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    labels=dec["input_ids"],
                )
        else:
            # Causal LM: concatenate prompt + response; labels mask out prompt tokens
            full_text = prompt + " " + response
            full_enc = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            prompt_enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels = full_enc["input_ids"].clone()
            labels[:, :prompt_len] = -100  # ignore prompt tokens in loss
            if full_enc["input_ids"].shape[1] == 0:
                return None
            with torch.no_grad():
                out = self.model(input_ids=full_enc["input_ids"], labels=labels)

        loss = min(out.loss.item(), 20.0)
        return math.exp(loss)

    def compute_perplexity_variance(self, prompts, responses):
        """Normalized PPL variance (CV) across prompt variants, clamped to [0, 1]."""
        import math
        ppl_values = [
            v for v in (
                self._compute_ppl(p, r)
                for p, r in zip(prompts, responses)
            ) if v is not None
        ]

        if len(ppl_values) < 2:
            return 0.0

        mean = sum(ppl_values) / len(ppl_values)
        if mean == 0:
            return 0.0

        std_dev = math.sqrt(sum((x - mean) ** 2 for x in ppl_values) / (len(ppl_values) - 1))
        cv = std_dev / mean
        return max(0.0, min(1.0, cv * 2.0))

    def compute_branching_factor(self, responses):
        entropies = []
        for resp in responses:
            inputs = self.tokenizer(resp, return_tensors="pt").to(self.device)
            if inputs["input_ids"].shape[1] == 0:
                continue
            with torch.no_grad():
                if self.is_seq2seq:
                    # Seq2Seq: use response as both encoder input and decoder labels
                    out = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        labels=inputs["input_ids"],
                    )
                else:
                    # Causal LM: standard next-token prediction
                    out = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                logits = out.logits.squeeze(0)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token_entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)
                entropies.append(token_entropy.mean().item())

        if not entropies:
            return 0.0
        avg_entropy = sum(entropies) / len(entropies)
        # Normalize: real LLMs rarely exceed entropy of 8.0 bits
        return max(0.0, min(1.0, avg_entropy / 8.0))
