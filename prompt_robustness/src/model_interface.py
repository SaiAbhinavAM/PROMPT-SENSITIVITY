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

# Known causal-LM families — skip AutoConfig round-trip for these (avoids
# an unnecessary gated-model auth hit and a ~1 s network call per model).
_KNOWN_CAUSAL = {
    "llama", "mistral", "qwen", "falcon", "gpt", "gemma", "phi", "bloom",
    "opt", "mpt", "stablelm", "deepseek", "baichuan", "internlm",
}


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _is_seq2seq(model_name: str) -> bool:
    """Return True if the model uses an encoder-decoder architecture."""
    name_lower = model_name.lower()
    # Fast-path: known causal families never need a network call.
    if any(k in name_lower for k in _KNOWN_CAUSAL):
        return False
    if any(k in name_lower for k in _SEQ2SEQ_ARCHS):
        return True
    try:
        cfg = AutoConfig.from_pretrained(model_name, token=_hf_token())
        arch = getattr(cfg, "model_type", "").lower()
        return any(k in arch for k in _SEQ2SEQ_ARCHS)
    except Exception:
        return False


class ModelInterface:
    def __init__(self, model_name: str, config: Config, quantization: str = None):
        self.config = config
        self.model_name = model_name
        self.quantization = quantization
        self.is_seq2seq = _is_seq2seq(model_name)

        token = _hf_token()
        if token is None:
            logger.warning(
                "HF_TOKEN not set — gated models (Llama, Mistral) will fail to download. "
                "Add HF_TOKEN to .env or set it as an environment variable."
            )

        # Device priority: CUDA (H100/A100) → MPS (Apple Silicon) → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"\n--- Loading: {model_name} ({'seq2seq' if self.is_seq2seq else 'causal'}) "
              f"on {self.device}{f'  quant={quantization}' if quantization else ''} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="left", token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Build load kwargs — device_map + torch_dtype on CUDA so large models
        # (including AWQ-quantized 70B) shard efficiently across GPU memory.
        load_kwargs: dict = {"token": token}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = "auto"

        # AWQ / GPTQ quantized models: pass the quantization config explicitly so
        # transformers uses the correct kernel (Marlin) instead of trying to load
        # the full fp16 weights (→ instant OOM on 70B).
        if quantization:
            self._apply_quantization_kwargs(load_kwargs, quantization)

        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
        self.model.eval()
        print(f"    ✓ {model_name} loaded.")

    @staticmethod
    def _apply_quantization_kwargs(load_kwargs: dict, quantization: str) -> None:
        """Wire the AWQ/GPTQ quantization spec into from_pretrained kwargs."""
        q = quantization.lower()
        if "awq" in q:
            try:
                # transformers ≥ 4.44: AwqConfig is the canonical path.
                from transformers import AwqConfig
                load_kwargs["quantization_config"] = AwqConfig(
                    bits=4,
                    backend="marlin" if "marlin" in q else "llm-awq",
                )
            except ImportError:
                # Fallback: autoawq direct loading (older transformers).
                try:
                    from awq import AutoAWQForCausalLM  # noqa: F401
                    # autoawq path — caller must use AutoAWQForCausalLM directly;
                    # we pop device_map because autoawq handles placement itself.
                    load_kwargs.pop("device_map", None)
                    load_kwargs.pop("torch_dtype", None)
                except ImportError:
                    logger.error(
                        "AWQ model requested but neither transformers.AwqConfig nor "
                        "autoawq is available. Install: pip install autoawq"
                    )
        elif "gptq" in q:
            try:
                from transformers import GPTQConfig
                load_kwargs["quantization_config"] = GPTQConfig(bits=4)
            except ImportError:
                logger.warning("GPTQConfig not available — loading without explicit quant config.")

    @property
    def _input_device(self):
        """Device to move input tensors to — cuda:0 for device_map=auto, else self.device."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return self.device

    def generate_responses(self, prompts: List[str]) -> List[str]:
        start_t = time.time()

        formatted_prompts = []
        for p in prompts:
            if "flan-t5" in self.model_name.lower() and not p.lower().startswith("summarize:"):
                formatted_prompts.append(f"summarize: {p}")
            elif hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                msgs = [{"role": "user", "content": p}]
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                )
            else:
                formatted_prompts.append(p)

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._input_device)

        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Only pass sampling knobs when sampling is enabled — passing temperature
        # with do_sample=False triggers transformers warnings and is a no-op.
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            torch.manual_seed(self.config.seed)  # reproducible sampling

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

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
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self._input_device)
            dec = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=512).to(self._input_device)
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
            full_enc = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(self._input_device)
            prompt_enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self._input_device)
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
            inputs = self.tokenizer(resp, return_tensors="pt").to(self._input_device)
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
