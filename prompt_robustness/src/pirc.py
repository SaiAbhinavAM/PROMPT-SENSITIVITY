"""
PIRC (Paraphrase-Invariant Residual Clamping) Forward Pass for LL-PIRC.

Orchestrates the full LL-PIRC pipeline:
1. Collect hidden states at ℓ* for all K paraphrases
2. Compute mean hidden state (consensus representation) across K variants
3. Register a forward hook that replaces anchor token positions with the
   mean hidden state during the primary prompt's forward pass
4. Run model.generate() with this hook active (greedy decoding)

Supports:
- Full clamping (α=1.0): h_clamped = mean_h
- Soft clamping (α<1.0): h_clamped = α * mean_h + (1-α) * h_original

Reference: method_analysis_prompt_sensitivity.md, Step 6 (lines 499-511,
           629-658)
"""

import torch
import logging
from typing import List, Tuple, Dict, Optional

from .logit_lens import LogitLensExtractor
from .sensitive_layer import SensitiveLayerDetector
from .anchor_tokens import AnchorTokenIdentifier

logger = logging.getLogger(__name__)


class PIRCGenerator:
    """
    Full LL-PIRC pipeline: sensitive layer detection → anchor token
    identification → clamped generation.

    At inference time, for K paraphrase variants of the same prompt:
    1. Extract hidden states at all layers to find ℓ*
    2. Identify anchor tokens at ℓ*
    3. Compute consensus hidden state at ℓ* across K variants
    4. Generate output with anchor positions clamped to consensus
    """

    def __init__(
        self,
        model,
        tokenizer,
        logit_lens: LogitLensExtractor,
        layer_detector: SensitiveLayerDetector,
        anchor_identifier: AnchorTokenIdentifier,
        alpha: float = 1.0,
        soft_clamp_alpha: float = 0.5,
        max_new_tokens: int = 200
    ):
        """
        Args:
            model: HuggingFace CausalLM model.
            tokenizer: HuggingFace tokenizer.
            logit_lens: LogitLensExtractor instance.
            layer_detector: SensitiveLayerDetector instance.
            anchor_identifier: AnchorTokenIdentifier instance.
            alpha: Clamping strength. 1.0 = full replacement, <1.0 = soft
                clamping (interpolation with original hidden state).
            soft_clamp_alpha: Alpha to use when soft clamping fallback is
                triggered (when ROUGE-L drops too much).
            max_new_tokens: Maximum tokens to generate.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logit_lens = logit_lens
        self.layer_detector = layer_detector
        self.anchor_identifier = anchor_identifier
        self.alpha = alpha
        self.soft_clamp_alpha = soft_clamp_alpha
        self.max_new_tokens = max_new_tokens

    def collect_hidden_at_layer(
        self,
        paraphrases: List[str],
        ell_star: int,
        device: Optional[str] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Collect hidden states at layer ℓ* for all K paraphrases.

        Handles variable sequence lengths by truncating to the minimum.

        Args:
            paraphrases: List of K prompt strings.
            ell_star: Target layer index.
            device: Device for computation.

        Returns:
            all_hidden: Tensor of shape (K, min_seq_len, d_model).
            min_len: The minimum sequence length used.
        """
        K = len(paraphrases)
        hidden_list = []
        seq_lengths = []

        for k, prompt in enumerate(paraphrases):
            hidden_states, _ = self.logit_lens.extract_all_hidden_states(
                self.tokenizer, prompt, device
            )
            h = hidden_states[ell_star]  # (seq_len, d_model)
            hidden_list.append(h)
            seq_lengths.append(h.shape[0])

        min_len = min(seq_lengths)

        # Truncate to min length and stack
        all_hidden = torch.stack(
            [h[:min_len] for h in hidden_list]
        )  # (K, min_len, d_model)

        logger.info(
            f"Collected hidden states at layer {ell_star}: "
            f"shape {all_hidden.shape}, seq_lengths={seq_lengths}"
        )

        return all_hidden, min_len

    def generate_with_clamping(
        self,
        primary_prompt: str,
        mean_h: torch.Tensor,
        anchor_mask: torch.Tensor,
        ell_star: int,
        alpha: Optional[float] = None,
        device: Optional[str] = None
    ) -> str:
        """
        Generate output with anchor positions clamped to the consensus
        hidden state at layer ℓ*.

        Registers a forward hook at layer ℓ* that replaces (or interpolates)
        anchor token hidden states with the mean hidden state across K
        paraphrases.

        Args:
            primary_prompt: The primary prompt to generate from.
            mean_h: Consensus hidden state, shape (min_seq_len, d_model).
            anchor_mask: Boolean mask, shape (min_seq_len - 1,) or
                (min_seq_len,). True at anchor positions.
            ell_star: The sensitive layer to apply clamping at.
            alpha: Clamping strength override. If None, uses self.alpha.
            device: Device for computation.

        Returns:
            generated_text: The generated output text (prompt stripped).
        """
        if device is None:
            device = next(self.model.parameters()).device

        if alpha is None:
            alpha = self.alpha

        # Ensure the anchor_mask aligns with hidden state dimensions.
        # The anchor_mask from the identifier is (min_seq_len - 1) because
        # PPL is computed for shifted positions. We need to map it to
        # hidden state positions. Position i in anchor_mask corresponds
        # to hidden state position i (predicting token i+1).
        # We'll create a full-length mask for the hidden state.
        num_anchor_positions = anchor_mask.shape[0]

        def clamp_hook(module, input, output):
            # In transformers 5.x with output_capturing the GPT-2 block
            # returns a plain Tensor during model.generate(), not a tuple.
            # Handle both cases gracefully.
            if isinstance(output, tuple):
                h = output[0]   # (batch, seq_len, d_model)
                rest = output[1:]
            else:
                h = output      # output IS the hidden state tensor
                rest = None

            seq_len = h.shape[1]

            # Only clamp positions that are within both the anchor mask
            # and the current sequence length
            clamp_len = min(num_anchor_positions, seq_len, mean_h.shape[0])

            if clamp_len > 0:
                mask = anchor_mask[:clamp_len]
                if mask.any():
                    mean_h_device = mean_h[:clamp_len].to(h.device).to(h.dtype)

                    if alpha >= 1.0:
                        # Full clamping
                        h[:, :clamp_len][:, mask] = mean_h_device[mask].unsqueeze(0)
                    else:
                        # Soft clamping: interpolation
                        h_original = h[:, :clamp_len][:, mask].clone()
                        h_mean = mean_h_device[mask].unsqueeze(0)
                        h[:, :clamp_len][:, mask] = (
                            alpha * h_mean + (1 - alpha) * h_original
                        )

            if rest is not None:
                return (h,) + rest
            else:
                return h

        # Prepare inputs
        primary_inputs = self.tokenizer(
            primary_prompt, return_tensors="pt"
        ).to(device)

        # Register hook at ℓ*
        target_layer = list(self.logit_lens.layers)[ell_star]
        hook = target_layer.register_forward_hook(clamp_hook)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **primary_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False  # greedy decoding for reproducibility
                )
        finally:
            hook.remove()

        # Strip prompt from output
        prompt_len = primary_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_text

    def generate_baseline(
        self,
        primary_prompt: str,
        device: Optional[str] = None,
    ) -> str:
        """
        Run model.generate() with NO clamping hook — i.e. the standard
        forward pass. Used by the smoke test to verify that PIRC clamping
        actually changes the output relative to the unintervened baseline.

        Uses the same greedy decoding and same max_new_tokens as
        generate_with_clamping for a fair comparison.
        """
        if device is None:
            device = next(self.model.parameters()).device

        primary_inputs = self.tokenizer(
            primary_prompt, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **primary_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy decoding — matches clamped run
            )

        prompt_len    = primary_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def run_pipeline(
        self,
        paraphrases: List[str],
        device: Optional[str] = None,
        tau_override: Optional[float] = None,
        alpha_override: Optional[float] = None
    ) -> Dict:
        """
        Run the full LL-PIRC pipeline end-to-end.

        Steps:
        1. Detect sensitive layer ℓ* via S(ℓ) curve
        2. Identify anchor tokens at ℓ*
        3. Collect hidden states and compute consensus
        4. Generate with clamping

        Args:
            paraphrases: List of K paraphrase prompt strings.
            device: Device for computation.
            tau_override: Optional tau override for anchor detection (retry).
            alpha_override: Optional alpha override for clamping strength.

        Returns:
            result: Dict with:
                - 'output': generated text
                - 'ell_star': sensitive layer index
                - 'S_curve': sensitivity curve dict
                - 'num_anchors': number of anchor positions
                - 'anchor_fraction': fraction of anchor positions
                - 'alpha_used': clamping alpha actually used
                - 'tau_used': anchor threshold actually used
        """
        # Step 1: Find sensitive layer
        logger.info("Step 1: Detecting sensitive layer...")
        ell_star, S_curve = self.layer_detector.detect(
            self.tokenizer, paraphrases, device
        )

        # Step 2: Identify anchor tokens
        logger.info(f"Step 2: Identifying anchor tokens at layer {ell_star}...")
        tau = tau_override if tau_override is not None else self.anchor_identifier.tau
        anchor_mask, anchor_stats = self.anchor_identifier.identify_anchors_with_threshold(
            self.tokenizer, paraphrases, ell_star, tau=tau, device=device
        )

        # Step 3: Collect hidden states and compute consensus
        logger.info(f"Step 3: Computing consensus hidden state at layer {ell_star}...")
        all_hidden, min_len = self.collect_hidden_at_layer(
            paraphrases, ell_star, device
        )
        mean_h = all_hidden.mean(dim=0)  # (min_len, d_model)

        # Step 4: Generate with clamping
        alpha = alpha_override if alpha_override is not None else self.alpha
        logger.info(
            f"Step 4: Generating with PIRC clamping "
            f"(α={alpha}, {anchor_stats['num_anchors']} anchors)..."
        )
        output_text = self.generate_with_clamping(
            primary_prompt=paraphrases[0],
            mean_h=mean_h,
            anchor_mask=anchor_mask,
            ell_star=ell_star,
            alpha=alpha,
            device=device
        )

        result = {
            'output': output_text,
            'ell_star': ell_star,
            'S_curve': {int(k): float(v) for k, v in S_curve.items()},
            'num_anchors': anchor_stats['num_anchors'],
            'anchor_fraction': anchor_stats['anchor_fraction'],
            'anchor_tokens_decoded': anchor_stats.get('anchor_tokens_decoded', []),
            'anchor_positions': anchor_stats.get('anchor_positions', []),
            'anchor_method': anchor_stats.get('method', 'unknown'),
            'alpha_used': alpha,
            'tau_used': tau,
        }

        logger.info(
            f"PIRC pipeline complete: ℓ*={ell_star}, "
            f"anchors={anchor_stats['num_anchors']}, "
            f"output_len={len(output_text)} chars"
        )

        return result
