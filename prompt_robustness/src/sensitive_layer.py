"""
Sensitive Layer Detector for LL-PIRC Pipeline.

For K paraphrase variants of the same prompt, computes the layer-wise
sensitivity signal S(ℓ) = Var_k[mean per-token PPL at layer ℓ], and
identifies the sensitive layer ℓ* via:
  1. Sharpest upward inflection in S(ℓ)  (primary method)
  2. Z-score threshold fallback: first ℓ where S(ℓ) > mean + z*std

Reference: method_analysis_prompt_sensitivity.md, Steps 3-4 (lines 486-493)
"""

import math
import torch
import logging
from typing import Dict, List, Tuple, Optional

from .logit_lens import LogitLensExtractor

logger = logging.getLogger(__name__)


class SensitiveLayerDetector:
    """
    Detects the sensitive layer ℓ* where prompt paraphrase sensitivity
    first manifests in the model's internal representations.

    The sensitivity signal S(ℓ) measures how much the model's per-layer
    perplexity varies across K paraphrase variants of the same prompt.
    High S(ℓ) indicates that the model's internal representations at
    layer ℓ are sensitive to surface-level prompt differences.
    """

    def __init__(
        self,
        logit_lens: LogitLensExtractor,
        scan_start_fraction: float = 0.25,
        scan_end_fraction: float = 1.0,
        method: str = "inflection",
        zscore_threshold: float = 2.0,
        force_ell_star: Optional[int] = None,
    ):
        """
        Args:
            logit_lens: LogitLensExtractor instance for the target model.
            scan_start_fraction: Start scanning from this fraction of total
                layers (skip early uninformative layers). Default 0.25 = L//4.
            scan_end_fraction: Scan up to this fraction. Default 1.0 = L.
            method: Detection method - "inflection" (argmax of ΔS) or
                "zscore" (first ℓ where S > mean + z*std).
            zscore_threshold: Z-score multiplier for the zscore method.
        """
        self.logit_lens = logit_lens
        self.num_layers = logit_lens.num_layers
        self.scan_start = int(self.num_layers * scan_start_fraction)
        self.scan_end = int(self.num_layers * scan_end_fraction)
        self.method = method
        self.zscore_threshold = zscore_threshold
        # Flaw §5.3 — ablation knob: when set, `detect()` returns this
        # layer instead of computing one. Lets us sweep ℓ* ∈ {6,9,12,18,
        # 24,28} and show layer 9 is empirically optimal (or not).
        self.force_ell_star = force_ell_star

        logger.info(
            f"SensitiveLayerDetector initialized: scanning layers "
            f"[{self.scan_start}, {self.scan_end}) of {self.num_layers} total. "
            f"Method: {self.method}"
            + (f" [force_ell_star={force_ell_star}]" if force_ell_star is not None else "")
        )

    def compute_sensitivity_curve(
        self,
        tokenizer,
        paraphrases: List[str],
        device: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Compute S(ℓ) = Var_k[mean per-token PPL at layer ℓ] across K
        paraphrase variants.

        For each paraphrase, extracts hidden states at all layers,
        computes per-token logit-lens PPL, then takes the mean across
        tokens. S(ℓ) is the variance of these K mean-PPL values at each
        layer.

        Args:
            tokenizer: HuggingFace tokenizer.
            paraphrases: List of K paraphrase prompt strings.
            device: Device for computation.

        Returns:
            S: Dict mapping layer_idx -> S(ℓ) (variance of mean PPL
                across paraphrases). Only includes layers in the scan range.
        """
        K = len(paraphrases)
        layer_range = (self.scan_start, self.scan_end)

        # Collect mean PPL per layer per paraphrase
        # all_mean_ppl[ell] = list of K mean-PPL values
        all_mean_ppl: Dict[int, List[float]] = {
            ell: [] for ell in range(self.scan_start, self.scan_end)
        }

        for k, prompt in enumerate(paraphrases):
            logger.debug(f"Processing paraphrase {k+1}/{K} for sensitivity curve")
            _, mean_ppl = self.logit_lens.compute_all_layer_ppl(
                tokenizer, prompt, device, layer_range
            )
            for ell, val in mean_ppl.items():
                if ell in all_mean_ppl:
                    all_mean_ppl[ell].append(val)

        # Compute S(ℓ) = Var_k[mean_PPL_k at layer ℓ].
        #
        # Robustness: even with the fp32 Logit Lens fix in place, individual
        # mean-PPL values can still be huge for very early layers (where the
        # unembedding head produces near-uniform distributions). We aggregate
        # in LOG space here: S(ℓ) = Var_k[ log(mean_PPL_k) ]. This:
        #   (a) is finite whenever mean_PPL is finite and > 0,
        #   (b) is scale-invariant (a 10× shift in PPL becomes an additive
        #       offset, which has zero contribution to variance),
        #   (c) ensures the inflection-of-S curve reflects RELATIVE changes
        #       across paraphrases, which is what the LL-PIRC story claims.
        S: Dict[int, float] = {}
        n_nonfinite_layers = 0
        for ell in range(self.scan_start, self.scan_end):
            if len(all_mean_ppl[ell]) != K:
                logger.warning(
                    f"Layer {ell}: expected {K} PPL values, got "
                    f"{len(all_mean_ppl[ell])}. Skipping."
                )
                continue

            vals = torch.tensor(all_mean_ppl[ell], dtype=torch.float64)
            # Take log of strictly-positive PPL values; non-finite or non-
            # positive entries are dropped from the variance computation so
            # one bad paraphrase does not destroy the whole curve.
            finite_pos = vals[(vals > 0) & torch.isfinite(vals)]
            if finite_pos.numel() < 2:
                # Cannot compute variance with <2 finite values; mark NaN so
                # the inflection detector can route around this layer.
                S[ell] = float("nan")
                n_nonfinite_layers += 1
                continue
            log_vals = torch.log(finite_pos)
            S[ell] = float(log_vals.var(unbiased=True).item())

        if n_nonfinite_layers:
            logger.warning(
                f"S(ℓ) has {n_nonfinite_layers}/{len(S)} non-finite layers — "
                "these will be skipped during ℓ* detection."
            )

        return S

    def find_sensitive_layer_inflection(
        self,
        S: Dict[int, float]
    ) -> int:
        """
        Find ℓ* as the layer with the sharpest upward inflection in S(ℓ).

        ℓ* = argmax_ℓ [S(ℓ) - S(ℓ-1)]

        NaN handling: ΔS values involving non-finite S(ℓ) are skipped
        entirely. If no finite ΔS remains (entire curve unusable), we fall
        through to the z-score fallback rather than silently picking the
        first scan layer — which was the pre-fix behaviour and the root
        cause of "ℓ* = scan_start+1 for every article" in the pilot results.

        Args:
            S: Sensitivity curve dict from compute_sensitivity_curve.

        Returns:
            ell_star: The sensitive layer index.
        """
        layers = sorted(S.keys())
        if len(layers) < 2:
            logger.warning("Not enough layers to compute inflection. Returning last layer.")
            return layers[-1] if layers else self.scan_end - 1

        # Compute finite differences ΔS(ℓ) = S(ℓ) - S(ℓ-1), skipping any
        # delta that depends on a non-finite endpoint.
        deltas = []
        for i in range(1, len(layers)):
            s_prev = S[layers[i - 1]]
            s_curr = S[layers[i]]
            if math.isfinite(s_prev) and math.isfinite(s_curr):
                deltas.append((layers[i], s_curr - s_prev))

        if not deltas:
            # Whole curve unusable — defer to z-score fallback, which itself
            # falls back to argmax(S) and ultimately to the last layer.
            logger.warning(
                "[inflection] All ΔS values were non-finite. Falling back to "
                "z-score detection."
            )
            return self.find_sensitive_layer_zscore(S)

        # Find the layer with the largest upward jump.
        ell_star, max_delta = max(deltas, key=lambda x: x[1])

        logger.info(
            f"Inflection method: ℓ* = {ell_star} "
            f"(ΔS = {max_delta:.6f}, {len(deltas)}/{len(layers)-1} finite deltas)"
        )
        return ell_star

    def find_sensitive_layer_zscore(
        self,
        S: Dict[int, float]
    ) -> int:
        """
        Find ℓ* as the first layer where S(ℓ) exceeds mean + z*std.

        Fallback method when inflection detection is noisy. Non-finite S(ℓ)
        values are excluded from both the threshold computation and the
        layer-scan candidates.

        Args:
            S: Sensitivity curve dict from compute_sensitivity_curve.

        Returns:
            ell_star: The sensitive layer index.
        """
        layers = sorted(S.keys())
        finite_pairs = [(ell, S[ell]) for ell in layers if math.isfinite(S[ell])]

        if len(finite_pairs) < 2:
            logger.warning(
                f"[zscore] Only {len(finite_pairs)} finite S(ℓ) values — "
                "falling back to last layer."
            )
            return layers[-1] if layers else self.scan_end - 1

        finite_layers = [p[0] for p in finite_pairs]
        vals = torch.tensor([p[1] for p in finite_pairs], dtype=torch.float64)

        mean_s = vals.mean().item()
        std_s = vals.std().item()
        threshold = mean_s + self.zscore_threshold * std_s

        logger.info(
            f"Z-score method: mean(S) = {mean_s:.6f}, std(S) = {std_s:.6f}, "
            f"threshold = {threshold:.6f}"
        )

        for ell, sv in finite_pairs:
            if sv > threshold:
                logger.info(f"Z-score method: ℓ* = {ell} (S = {sv:.6f})")
                return ell

        # If no layer exceeds threshold, fall back to argmax over finite vals.
        ell_star, _ = max(finite_pairs, key=lambda p: p[1])
        logger.warning(
            f"No layer exceeded z-score threshold. "
            f"Falling back to argmax(S): ℓ* = {ell_star}"
        )
        return ell_star

    def find_sensitive_layer(
        self,
        S: Dict[int, float]
    ) -> int:
        """
        Find the sensitive layer ℓ* using the configured method.

        Args:
            S: Sensitivity curve dict from compute_sensitivity_curve.

        Returns:
            ell_star: The sensitive layer index.
        """
        if self.method == "inflection":
            return self.find_sensitive_layer_inflection(S)
        elif self.method == "zscore":
            return self.find_sensitive_layer_zscore(S)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'inflection' or 'zscore'."
            )

    def detect(
        self,
        tokenizer,
        paraphrases: List[str],
        device: Optional[str] = None
    ) -> Tuple[int, Dict[int, float]]:
        """
        Full detection pipeline: compute sensitivity curve and find ℓ*.

        Args:
            tokenizer: HuggingFace tokenizer.
            paraphrases: List of K paraphrase prompt strings.
            device: Device for computation.

        Returns:
            ell_star: The identified sensitive layer.
            S: The full sensitivity curve dict.
        """
        S = self.compute_sensitivity_curve(tokenizer, paraphrases, device)
        if self.force_ell_star is not None:
            ell_star = int(self.force_ell_star)
            logger.info(
                f"[ablation] ℓ* forced to {ell_star} (skipping detection)"
            )
        else:
            ell_star = self.find_sensitive_layer(S)
        return ell_star, S
