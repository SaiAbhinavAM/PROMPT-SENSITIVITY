"""
llm_backend.py — shared causal instruct-LLM backend for scoring (judge + NLI).

On the H100, the judge and the faithfulness scorer use the SAME large model
(e.g. Llama-3.x-70B-AWQ, independent of the 8B subjects). This module loads it
ONCE (cached by model id) and exposes a simple batched chat interface, so both
scorers share a single resident model.

Backend selection (env LLM_BACKEND): "auto" (default) tries vLLM, falls back to
HF transformers; "vllm" / "hf" force one. All heavy imports are lazy so this
module imports fine on a laptop without vllm/torch-cuda.
"""

import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, "ChatLLM"] = {}


class ChatLLM:
    def __init__(
        self,
        model_id: str,
        backend: str = None,
        quantization: Optional[str] = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ):
        self.model_id = model_id
        self.backend = backend or os.getenv("LLM_BACKEND", "auto")
        self.quantization = quantization or os.getenv("LLM_QUANTIZATION") or None
        self._impl = None  # "vllm" or "hf"

        if self.backend in ("auto", "vllm"):
            try:
                self._init_vllm(max_model_len, gpu_memory_utilization)
                self._impl = "vllm"
            except Exception as e:  # pragma: no cover - env dependent
                if self.backend == "vllm":
                    raise
                logger.info(f"vLLM unavailable ({e}); falling back to HF transformers")
        if self._impl is None:
            self._init_hf()
            self._impl = "hf"
        logger.info(f"ChatLLM ready: {model_id} via {self._impl}")

    def _init_vllm(self, max_model_len, gpu_memory_utilization):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = dict(
            model=self.model_id, tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, dtype="auto", trust_remote_code=True,
        )
        if self.quantization:
            kwargs["quantization"] = self.quantization
        self._llm = LLM(**kwargs)
        self._SamplingParams = SamplingParams

    def _init_hf(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()

    def chat_batch(self, message_lists: List[List[Dict]], max_new_tokens: int = 8) -> List[str]:
        """Greedy chat completion for a batch of message lists → list of strings."""
        prompts = [
            self._tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in message_lists
        ]
        if self._impl == "vllm":
            sp = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            outs = self._llm.generate(prompts, sp)
            return [o.outputs[0].text.strip() for o in outs]

        # HF path
        torch = self._torch
        results = []
        for p in prompts:
            inputs = self._tok(p, return_tensors="pt", truncation=True, max_length=4096).to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tok.pad_token_id,
                )
            text = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(text.strip())
        return results


def get_chat_llm(model_id: str, **kwargs) -> ChatLLM:
    """Return a cached ChatLLM for model_id (loads once; shared across scorers)."""
    if model_id not in _CACHE:
        _CACHE[model_id] = ChatLLM(model_id, **kwargs)
    return _CACHE[model_id]
