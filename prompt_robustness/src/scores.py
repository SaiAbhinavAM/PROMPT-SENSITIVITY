"""
scores.py — canonical composite-score formulas (single source of truth).

Both the live evaluator (src/evaluator.py) and the offline re-aggregator
(reaggregate.py) import these so the PRI/ORI/IFI/Final definitions can never
drift apart. If a formula changes, change it HERE (and log it in method.md).

All scalar; callers vectorize via apply if needed.
"""

# --- Canonical PRI composition (R3): three non-collinear axes ---
PRI_W_CONSISTENCY = 0.40   # SMS
PRI_W_QUALITY = 0.35       # CS vs reference
PRI_W_FAITHFULNESS = 0.25  # NLI entailment

# Structural gate (R4): degenerate too-short outputs are penalized once.
SHORT_OUTPUT_LEN = 12
SHORT_OUTPUT_PENALTY = 0.85

# Final score (static weighting).
FINAL_W_PRI = 0.60
FINAL_W_HUMAN = 0.40


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_pri(
    consistency: float,
    quality: float,
    faithfulness: float,
    avg_length: float,
    w_consistency: float = PRI_W_CONSISTENCY,
    w_quality: float = PRI_W_QUALITY,
    w_faithfulness: float = PRI_W_FAITHFULNESS,
) -> float:
    """PRI = w_c·consistency + w_q·quality + w_f·faithfulness, short-output gated."""
    pri = (w_consistency * clamp01(consistency)
           + w_quality * clamp01(quality)
           + w_faithfulness * clamp01(faithfulness))
    if avg_length < SHORT_OUTPUT_LEN:
        pri *= SHORT_OUTPUT_PENALTY
    return clamp01(pri)


def compute_ori(sms: float, auc_e: float, trd: float, kpig: float) -> float:
    """Observable Robustness Index (TRD is lower-is-better → 1-trd)."""
    return clamp01((sms + auc_e + (1.0 - trd) + kpig) / 4.0)


def compute_ifi(ppl_var: float, bf: float) -> float:
    """Intrinsic Fidelity Index (intra-model diagnostic only)."""
    return clamp01(1.0 - (ppl_var + bf) / 2.0)


def compute_final_static(
    pri: float,
    human_score: float,
    w_pri: float = FINAL_W_PRI,
    w_human: float = FINAL_W_HUMAN,
) -> float:
    """Static Final_Score = w_pri·PRI + w_human·Human."""
    return clamp01(w_pri * pri + w_human * human_score)
