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

# Diagnostic dual-pillar score (publication framing): the primary PRI remains
# the fair cross-model ranking score above; this score is for causal diagnosis.
DIAG_W_ORI = 0.50
DIAG_W_IFI = 0.50
DIAGNOSIS_THRESHOLD = 0.70
HARMONIC_EPS = 1e-8


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


def weighted_harmonic_mean(values, weights=None, eps: float = HARMONIC_EPS) -> float:
    """Weighted harmonic mean over [0, 1] scores.

    This aggregation is intentionally strict for diagnostic use: if any pillar
    approaches zero, the composite also approaches zero.
    """
    vals = [clamp01(v) for v in values]
    if not vals:
        return 0.0

    if weights is None:
        wts = [1.0] * len(vals)
    else:
        wts = [float(w) for w in weights]
        if len(wts) != len(vals):
            raise ValueError("weights length must match values length")

    total_weight = sum(wts)
    if total_weight <= 0:
        return 0.0
    if any(v <= 0.0 for v in vals):
        return 0.0

    denom = sum(w / max(v, eps) for v, w in zip(vals, wts))
    return clamp01(total_weight / denom) if denom > 0 else 0.0


def compute_diagnostic_ori(
    sms: float,
    auc_e: float,
    trd: float,
    kpig: float,
) -> float:
    """Diagnostic Observable Robustness Index via harmonic mean.

    Inputs are normalized so higher is better. TRD is lower-is-better and is
    inverted before synthesis.
    """
    return weighted_harmonic_mean([
        sms,
        auc_e,
        1.0 - clamp01(trd),
        kpig,
    ])


def compute_diagnostic_ifi(
    ppl_var: float,
    bf: float,
    pc_stab: float = None,
) -> float:
    """Diagnostic Intrinsic Fidelity Index via harmonic mean.

    PPL variance, branching factor instability, and optional prompt-confidence
    stability are lower-is-better, so they are inverted before synthesis.
    """
    components = [
        1.0 - clamp01(ppl_var),
        1.0 - clamp01(bf),
    ]
    if pc_stab is not None:
        components.append(1.0 - clamp01(pc_stab))
    return weighted_harmonic_mean(components)


def compute_diagnostic_pri(
    diagnostic_ori: float,
    diagnostic_ifi: float,
    w_ori: float = DIAG_W_ORI,
    w_ifi: float = DIAG_W_IFI,
) -> float:
    """Dual-pillar Diagnostic PRI = HM(ORI, IFI)."""
    return weighted_harmonic_mean(
        [diagnostic_ori, diagnostic_ifi],
        [w_ori, w_ifi],
    )


def compute_dual_pillar_diagnosis(
    diagnostic_ori: float,
    diagnostic_ifi: float,
    threshold: float = DIAGNOSIS_THRESHOLD,
) -> str:
    """Classify a sample using the 2x2 ORI x IFI diagnosis matrix."""
    high_ori = clamp01(diagnostic_ori) >= threshold
    high_ifi = clamp01(diagnostic_ifi) >= threshold

    if high_ori and high_ifi:
        return "Robust"
    if high_ori and not high_ifi:
        return "Externally Stable / Internally Fragile"
    if not high_ori and high_ifi:
        return "Internally Stable / Output-Sensitive"
    return "Fragile"
