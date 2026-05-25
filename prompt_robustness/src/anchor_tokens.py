"""
Anchor Token Identifier for LL-PIRC Pipeline.
=============================================

At the sensitive layer ℓ*, identifies anchor tokens — positions where the
model is both confident (low PPL) AND consistently confident across all K
paraphrase variants (low PPL variance).

This module uses a RELATIVE PERCENTILE-BASED criterion (model-agnostic, can
never return zero anchors) rather than absolute PPL thresholds. The previous
absolute dual-threshold approach (mean PPL < τ AND var PPL < τ_var) was
brittle: different models have wildly different PPL scales, and a poorly
tuned τ silently produced 0 anchors → PIRC clamping became a no-op.

Selection algorithm (NEW DEFAULT):
    1. Compute per-token mean PPL and var PPL across K variants at ℓ*
    2. Rank tokens by combined stability score:
           stability_rank = rank(mean_ppl) + rank(var_ppl)
           (lower rank = more stable = better anchor candidate)
    3. Select the bottom P% most-stable tokens as anchors (default P=30)
    4. Filter out pure punctuation + common stopwords (if filter leaves
       ≥ 1 anchor; otherwise fall back to unfiltered selection)
    5. GUARANTEE: at least 1 anchor, at most ⌊min_len/2⌋ anchors

The legacy absolute-threshold method is preserved as
`identify_anchors_threshold` for backward compatibility / ablation studies.

Reference: method_analysis_prompt_sensitivity.md, Step 5
"""

import string
import torch
import logging
from typing import List, Tuple, Optional, Set

from .logit_lens import LogitLensExtractor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Common English stopwords (lowercase, stripped, no punctuation).
# Kept minimal so we don't kill all anchors on short prompts.
# ─────────────────────────────────────────────────────────────
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in",
    "on", "at", "by", "for", "with", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "doing", "have", "has",
    "had", "having", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "them", "us", "him", "her", "my", "your",
    "his", "their", "our", "its", "from", "into", "out", "up", "down",
    "so", "than", "then", "too", "very", "can", "will", "just", "not",
    "no", "nor", "only",
}


class AnchorTokenIdentifier:
    """
    Identifies anchor token positions at a given layer via percentile-based
    ranking of per-token logit-lens PPL statistics across paraphrase variants.

    Args:
        logit_lens          : LogitLensExtractor instance.
        anchor_percentile   : Select the bottom P% most-stable tokens
                              (default 30). 0 < P ≤ 100.
        filter_content_words: If True, exclude punctuation-only and
                              stopword tokens from anchor selection.
                              Falls back to unfiltered if filtering removes
                              all candidates.
        tau                 : (LEGACY) PPL threshold for the absolute method,
                              used only by `identify_anchors_threshold`.
        tau_var             : (LEGACY) PPL variance threshold for the
                              absolute method.
    """

    def __init__(
        self,
        logit_lens:           LogitLensExtractor,
        anchor_percentile:    float = 30.0,
        filter_content_words: bool  = True,
        tau:                  float = 3.0,
        tau_var:              float = 1.0,
    ):
        assert 0 < anchor_percentile <= 100, "anchor_percentile must be in (0, 100]"
        self.logit_lens           = logit_lens
        self.anchor_percentile    = float(anchor_percentile)
        self.filter_content_words = bool(filter_content_words)
        self.tau                  = tau
        self.tau_var              = tau_var

        logger.info(
            f"AnchorTokenIdentifier initialised: "
            f"percentile={self.anchor_percentile}%, "
            f"filter_content_words={self.filter_content_words} "
            f"(legacy thresholds τ={tau}, τ_var={tau_var} kept for fallback)"
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_ppl_across_variants(
        logit_lens:  LogitLensExtractor,
        tokenizer,
        paraphrases: List[str],
        ell_star:    int,
        device:      Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Extract per-token PPL at layer ℓ* for K paraphrases, aligned to the
        minimum sequence length.

        Returns:
            ppl_stack : (K, min_len)
            mean_ppl  : (min_len,)
            var_ppl   : (min_len,)
            min_len   : int
        """
        K = len(paraphrases)
        ppl_per_token_all = []
        seq_lengths       = []

        for k, prompt in enumerate(paraphrases):
            hidden_states, input_ids = logit_lens.extract_all_hidden_states(
                tokenizer, prompt, device
            )
            ppl_tok = logit_lens.compute_per_token_ppl(
                hidden_states[ell_star], input_ids
            )
            ppl_per_token_all.append(ppl_tok)
            seq_lengths.append(len(ppl_tok))
            logger.debug(
                f"Paraphrase {k+1}/{K}: seq_len={len(ppl_tok)+1}, "
                f"mean_ppl={ppl_tok.mean().item():.3f}"
            )

        min_len = min(seq_lengths)
        logger.info(
            f"Sequence lengths across {K} variants: {seq_lengths}. "
            f"Using min_len={min_len} for alignment."
        )

        ppl_stack = torch.stack(
            [ppl[:min_len] for ppl in ppl_per_token_all]
        )  # (K, min_len)

        mean_ppl = ppl_stack.mean(dim=0)   # (min_len,)
        var_ppl  = ppl_stack.var(dim=0)    # (min_len,)
        return ppl_stack, mean_ppl, var_ppl, min_len

    @staticmethod
    def _is_content_token(decoded: str) -> bool:
        """
        Return True if a decoded token is a 'content' token, i.e. NOT
        pure punctuation/whitespace and NOT a common stopword.
        """
        stripped = decoded.strip()
        if not stripped:
            return False
        # All characters are punctuation → reject
        if all(c in string.punctuation for c in stripped):
            return False
        # Lowercased stripped form is in stopword list → reject
        if stripped.lower() in _STOPWORDS:
            return False
        return True

    # ── Public API: percentile-based anchor selection (NEW DEFAULT) ───────

    def identify_anchors(
        self,
        tokenizer,
        paraphrases: List[str],
        ell_star:    int,
        device:      Optional[str] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Identify anchor positions via percentile-based stability ranking.

        Returns:
            anchor_mask : Boolean tensor (min_len,), True at anchor positions.
            stats       : Dict with diagnostics:
                - 'mean_ppl', 'var_ppl', 'ppl_stack', 'min_seq_len'
                - 'num_anchors', 'anchor_fraction', 'anchor_positions'
                - 'anchor_tokens_decoded' : list[str] (decoded anchor tokens)
                - 'stability_score'       : combined rank-sum per position
                - 'filtering_applied'     : bool (was content-word filter
                                                 actually applied?)
                - 'method'                : 'percentile'
        """
        K = len(paraphrases)
        ppl_stack, mean_ppl, var_ppl, min_len = self._extract_ppl_across_variants(
            self.logit_lens, tokenizer, paraphrases, ell_star, device
        )

        if min_len == 0:
            logger.warning("min_len = 0 — cannot identify anchors.")
            empty_mask = torch.zeros(0, dtype=torch.bool)
            return empty_mask, {
                "mean_ppl": mean_ppl, "var_ppl": var_ppl, "ppl_stack": ppl_stack,
                "min_seq_len": 0, "num_anchors": 0, "anchor_fraction": 0.0,
                "anchor_positions": [], "anchor_tokens_decoded": [],
                "stability_score": torch.zeros(0), "filtering_applied": False,
                "method": "percentile",
            }

        # ── Compute stability ranks (lower = more stable) ─────────────────
        mean_ranks = torch.argsort(torch.argsort(mean_ppl))  # 0 = smallest
        var_ranks  = torch.argsort(torch.argsort(var_ppl))
        stability  = (mean_ranks + var_ranks).float()        # (min_len,)

        # ── Decode token strings at primary prompt positions ──────────────
        # We need the primary prompt's input_ids to decode anchor tokens.
        primary_inputs = tokenizer(paraphrases[0], return_tensors="pt")
        primary_ids    = primary_inputs["input_ids"][0]
        # The PPL is computed for positions 1..N-1 (predicting next token);
        # so position i in our stats array corresponds to the i-th *predicted*
        # token in primary_ids. We decode primary_ids[i+1] when available.
        decoded_per_pos: List[str] = []
        for i in range(min_len):
            tok_id_idx = i + 1 if (i + 1) < len(primary_ids) else i
            tok_id     = int(primary_ids[tok_id_idx].item())
            decoded_per_pos.append(tokenizer.decode([tok_id]))

        # ── Determine target number of anchors (enforce bounds) ───────────
        n_target = int(round(min_len * self.anchor_percentile / 100.0))
        n_target = max(1, min(n_target, min_len // 2 if min_len >= 2 else 1))

        # ── Sort positions by ascending stability score ───────────────────
        sorted_positions = torch.argsort(stability).tolist()

        selected: List[int] = []
        filtering_applied = False

        if self.filter_content_words:
            for pos in sorted_positions:
                if len(selected) >= n_target:
                    break
                if self._is_content_token(decoded_per_pos[pos]):
                    selected.append(pos)
            if len(selected) == 0:
                # Fallback: filter killed everything — use raw ranking
                logger.warning(
                    "Content-word filter removed all anchor candidates — "
                    "falling back to unfiltered percentile selection."
                )
                selected = sorted_positions[:n_target]
                filtering_applied = False
            else:
                filtering_applied = True
                # If filter selected fewer than n_target (very short prompts
                # with mostly stopwords), top up with unfiltered candidates
                if len(selected) < n_target:
                    for pos in sorted_positions:
                        if pos in selected:
                            continue
                        selected.append(pos)
                        if len(selected) >= n_target:
                            break
        else:
            selected = sorted_positions[:n_target]
            filtering_applied = False

        # ── Build mask ────────────────────────────────────────────────────
        anchor_mask = torch.zeros(min_len, dtype=torch.bool)
        for pos in selected:
            anchor_mask[pos] = True

        num_anchors    = int(anchor_mask.sum().item())
        anchor_frac    = num_anchors / min_len if min_len > 0 else 0.0
        anchor_positions      = sorted(selected)
        anchor_tokens_decoded = [decoded_per_pos[p] for p in anchor_positions]

        # ── Log decoded anchors with stats ────────────────────────────────
        logger.info(
            f"Anchors @ layer {ell_star}: {num_anchors}/{min_len} "
            f"({anchor_frac:.1%}) — method=percentile P={self.anchor_percentile}%, "
            f"content_word_filter={filtering_applied}"
        )
        for pos in anchor_positions[:20]:  # cap log spam at 20
            logger.info(
                f"  pos={pos:3d}  tok={decoded_per_pos[pos]!r:20s}  "
                f"mean_ppl={mean_ppl[pos].item():.3f}  "
                f"var_ppl={var_ppl[pos].item():.4f}  "
                f"stability={stability[pos].item():.1f}"
            )
        if len(anchor_positions) > 20:
            logger.info(f"  … ({len(anchor_positions) - 20} more anchors)")

        stats = {
            "mean_ppl":              mean_ppl,
            "var_ppl":               var_ppl,
            "ppl_stack":             ppl_stack,
            "min_seq_len":           min_len,
            "num_anchors":           num_anchors,
            "anchor_fraction":       anchor_frac,
            "anchor_positions":      anchor_positions,
            "anchor_tokens_decoded": anchor_tokens_decoded,
            "stability_score":       stability,
            "filtering_applied":     filtering_applied,
            "method":                "percentile",
        }
        return anchor_mask, stats

    # ── LEGACY: absolute-threshold method (kept for ablations) ────────────

    def identify_anchors_threshold(
        self,
        tokenizer,
        paraphrases: List[str],
        ell_star:    int,
        device:      Optional[str] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        LEGACY absolute-threshold method:
            anchor_mask = (mean_ppl < τ) & (var_ppl < τ_var)

        Can silently return 0 anchors when thresholds are mistuned —
        prefer `identify_anchors` (percentile) for production use.
        """
        ppl_stack, mean_ppl, var_ppl, min_len = self._extract_ppl_across_variants(
            self.logit_lens, tokenizer, paraphrases, ell_star, device
        )

        anchor_mask = (mean_ppl < self.tau) & (var_ppl < self.tau_var)
        num_anchors = int(anchor_mask.sum().item())
        frac        = num_anchors / min_len if min_len > 0 else 0.0

        logger.info(
            f"[LEGACY threshold] Anchors @ layer {ell_star}: "
            f"{num_anchors}/{min_len} ({frac:.1%}) "
            f"(τ={self.tau}, τ_var={self.tau_var})"
        )

        return anchor_mask, {
            "mean_ppl":         mean_ppl,
            "var_ppl":          var_ppl,
            "ppl_stack":        ppl_stack,
            "min_seq_len":      min_len,
            "num_anchors":      num_anchors,
            "anchor_fraction":  frac,
            "method":           "threshold",
        }

    def identify_anchors_with_threshold(
        self,
        tokenizer,
        paraphrases: List[str],
        ell_star:    int,
        tau:         float,
        tau_var:     Optional[float] = None,
        device:      Optional[str]   = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Back-compat shim used by `pirc.py`'s retry logic. Sets τ / τ_var and
        delegates to the percentile-based identifier (which is robust), so
        the retry semantics still work but never produce 0 anchors.
        """
        # Preserve the call signature, but route to percentile selection.
        original_tau     = self.tau
        original_tau_var = self.tau_var
        self.tau         = tau
        if tau_var is not None:
            self.tau_var = tau_var
        try:
            return self.identify_anchors(tokenizer, paraphrases, ell_star, device)
        finally:
            self.tau     = original_tau
            self.tau_var = original_tau_var
