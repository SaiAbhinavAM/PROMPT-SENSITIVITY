"""
Logit Lens Extractor for LL-PIRC Pipeline.

Implements the Logit Lens mechanism: at each decoder layer ℓ, applies the
final layer norm + lm_head to get per-layer logits, then computes per-token
perplexity. This enables per-layer sensitivity analysis across prompt
paraphrases.

Reference: method_analysis_prompt_sensitivity.md, Section Q1 (lines 260-308)
           Belrose et al. "Eliciting Latent Predictions from Transformers
           with the Tuned Lens" (arXiv 2303.08112, ICML 2023)
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Dict, Tuple, Optional


class LogitLensExtractor:
    """
    Extracts per-layer logits and perplexity using the Logit Lens technique.

    The Logit Lens projects intermediate hidden states h^(ℓ) through the
    model's final layer norm and unembedding head to obtain token probability
    distributions at each layer:

        logits^(ℓ) = lm_head(layer_norm(h^(ℓ)))

    From these logits, per-token perplexity is computed as:
        PPL(t_i) = exp(-log_softmax(logits^(ℓ))[t_{i+1}])
    """

    def __init__(self, model, cast_to_float32: bool = True):
        """
        Args:
            model: HuggingFace CausalLM model (e.g., LlamaForCausalLM).
            cast_to_float32: If True, cast hidden states to float32 before
                computing logits for numerical stability (important for fp16
                models).
        """
        self.model = model
        self.cast_to_float32 = cast_to_float32

        # Identify the final layer norm and lm_head.
        # Supports LLaMA-style (model.model.norm) and GPT-2-style
        # (model.transformer.ln_f) architectures.
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            # LLaMA / Mistral / Qwen style
            self.final_norm = model.model.norm
            self.lm_head = model.lm_head
            self.layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            # GPT-2 / GPT-Neo style
            self.final_norm = model.transformer.ln_f
            self.lm_head = model.lm_head
            self.layers = model.transformer.h
        else:
            raise ValueError(
                f"Unsupported model architecture: {type(model).__name__}. "
                "Expected LLaMA-style (model.model.norm) or GPT-2-style "
                "(model.transformer.ln_f)."
            )

        self.num_layers = len(self.layers)

    def compute_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply the Logit Lens: final layer norm + lm_head projection.

        Args:
            hidden_state: Tensor of shape (seq_len, d_model) or
                (batch, seq_len, d_model).

        Returns:
            logits: Tensor of shape (..., seq_len, vocab_size).
        """
        if self.cast_to_float32:
            hidden_state = hidden_state.float()

        # Apply final layer norm
        h_normed = self.final_norm(hidden_state)
        # Project through unembedding matrix
        logits = self.lm_head(h_normed)
        return logits

    def compute_per_token_ppl(
        self,
        hidden_state: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token perplexity at a given layer using the Logit Lens.

        Args:
            hidden_state: Tensor of shape (seq_len, d_model). The hidden
                state at a specific layer for a single sequence.
            labels: Tensor of shape (seq_len,). The token IDs of the input
                sequence.

        Returns:
            ppl_per_token: Tensor of shape (seq_len - 1,). Per-token
                perplexity values. ppl_per_token[i] is the perplexity of
                predicting labels[i+1] from hidden_state[i].
        """
        logits = self.compute_logits(hidden_state)

        # Compute log-softmax over vocabulary dimension
        # logits[:-1] predicts tokens labels[1:]
        log_probs = F.log_softmax(logits[:-1], dim=-1)  # (seq_len-1, vocab)

        # Gather the log-probability of the actual next token
        token_log_probs = log_probs.gather(
            -1, labels[1:].unsqueeze(-1)
        ).squeeze(-1)  # (seq_len-1,)

        # Perplexity = exp(-log_prob)
        ppl_per_token = torch.exp(-token_log_probs)
        return ppl_per_token

    @contextmanager
    def hook_all_layers(self):
        """
        Context manager that registers forward hooks on all decoder layers
        to capture hidden states.

        Usage:
            with extractor.hook_all_layers() as hidden_states:
                model(**inputs)
            # hidden_states[layer_idx] = Tensor(batch, seq_len, d_model)

        Yields:
            hidden_states: Dict[int, Tensor] mapping layer index to captured
                hidden state.
        """
        hidden_states: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # output[0] is the hidden state tensor for decoder layers
                h = output[0].detach()
                if self.cast_to_float32:
                    h = h.float()
                hidden_states[layer_idx] = h
            return hook_fn

        try:
            for i, layer in enumerate(self.layers):
                hooks.append(layer.register_forward_hook(make_hook(i)))
            yield hidden_states
        finally:
            for h in hooks:
                h.remove()

    def extract_all_hidden_states(
        self,
        tokenizer,
        prompt: str,
        device: Optional[str] = None
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Run a forward pass with hooks to extract hidden states at all layers.

        Args:
            tokenizer: HuggingFace tokenizer.
            prompt: Input text string.
            device: Device to place inputs on. If None, uses model's device.

        Returns:
            hidden_states: Dict mapping layer_idx -> Tensor(seq_len, d_model).
                Hidden states are squeezed to remove batch dimension.
            input_ids: Tensor(seq_len,). The tokenized input IDs.
        """
        if device is None:
            device = next(self.model.parameters()).device

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with self.hook_all_layers() as hidden_states:
            with torch.no_grad():
                self.model(**inputs)

        # Squeeze batch dimension (batch=1)
        squeezed = {k: v.squeeze(0) for k, v in hidden_states.items()}
        input_ids = inputs["input_ids"].squeeze(0)

        return squeezed, input_ids

    def compute_all_layer_ppl(
        self,
        tokenizer,
        prompt: str,
        device: Optional[str] = None,
        layer_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        """
        Compute per-token PPL at every layer (or a subset) for a single prompt.

        Args:
            tokenizer: HuggingFace tokenizer.
            prompt: Input text string.
            device: Device to place inputs on.
            layer_range: Optional (start, end) layer indices to compute PPL
                for. If None, computes for all layers.

        Returns:
            ppl_per_token: Dict mapping layer_idx -> Tensor(seq_len-1,)
                with per-token PPL.
            mean_ppl: Dict mapping layer_idx -> float with mean per-token PPL.
        """
        hidden_states, input_ids = self.extract_all_hidden_states(
            tokenizer, prompt, device
        )

        start = layer_range[0] if layer_range else 0
        end = layer_range[1] if layer_range else self.num_layers

        ppl_per_token = {}
        mean_ppl = {}

        for ell in range(start, end):
            if ell in hidden_states:
                ppl = self.compute_per_token_ppl(hidden_states[ell], input_ids)
                ppl_per_token[ell] = ppl
                mean_ppl[ell] = ppl.mean().item()

        return ppl_per_token, mean_ppl
