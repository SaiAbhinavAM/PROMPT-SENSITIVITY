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
#
# Flaw §2.3 clarification: the arithmetic PRI (above) and the harmonic
# Diagnostic_PRI (below) DISAGREE on model rank (Pearson r ≈ -0.13 in the
# 5-instance audit). They measure different constructs by design — arithmetic
# rewards balanced moderates, harmonic punishes any one weak axis. Publication
# policy:
#   * `pri` (arithmetic)             — PRIMARY ranking score in main tables.
#   * `diagnostic_pri` (harmonic HM) — APPENDIX-ONLY strict-evaluation
#                                       companion. Never ship the two as if
#                                       they measure the same thing.
DIAG_W_ORI = 0.50
DIAG_W_IFI = 0.50
DIAGNOSIS_THRESHOLD = 0.70
HARMONIC_EPS = 1e-8

# ---------------------------------------------------------------------------
# Diagnostic sub-component weights (Phase-6 — empirically motivated by the
# FLAWS_AND_FIXES.pdf §3 collinearity audit on the 5-instance run).
#
# Diagnostic_ORI = HM(SMS, AUC-E, 1-TRD, KPIG)
#   The pre-Phase-6 default was uniform (1.0 each), which let the
#   SMS↔TRD r = -0.98 redundancy double-count consistency. The new weights
#   demote TRD to a tie-breaker and reward the two independent axes
#   (SMS and AUC-E) that carry most of the legitimate signal.
#
#   SMS    — primary semantic-stability signal           → 0.40 (HIGHEST)
#   AUC-E  — performance elasticity across variants      → 0.30
#   KPIG   — reference-coverage axis (independent of SMS)→ 0.20
#   1-TRD  — collinear with SMS at r=-0.98 (§3.7)        → 0.10 (LOWEST)
#
# Diagnostic_IFI = HM(1-PPL_var, 1-BF [, 1-PC_stab])
#   PPL variance is the cleanest intra-model stability signal; branching-
#   factor is noisier and second-order. PC_stab, when present, is a small
#   additional confidence-based signal.
#
#   PPL_var — primary intra-model stability               → 0.50 (HIGHEST)
#   BF      — secondary; noisier, entropy-derived         → 0.30
#   PC_stab — optional prompt-confidence stability        → 0.20
#
# Override any of these via Config (env vars DIAG_*) for ablations — the
# pre-Phase-6 equal-weight behaviour is recoverable by passing all 1.0.
# ---------------------------------------------------------------------------
DIAG_ORI_W_SMS   = 0.40
DIAG_ORI_W_AUC_E = 0.30
DIAG_ORI_W_KPIG  = 0.20
DIAG_ORI_W_TRD   = 0.10

DIAG_IFI_W_PPL_VAR = 0.50
DIAG_IFI_W_BF      = 0.30
DIAG_IFI_W_PC_STAB = 0.20


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
    w_sms:   float = DIAG_ORI_W_SMS,
    w_auc_e: float = DIAG_ORI_W_AUC_E,
    w_kpig:  float = DIAG_ORI_W_KPIG,
    w_trd:   float = DIAG_ORI_W_TRD,
) -> float:
    """Diagnostic Observable Robustness Index via weighted harmonic mean.

    Inputs are normalized so higher is better. TRD is lower-is-better and is
    inverted before synthesis. Default weights (0.40/0.30/0.20/0.10 for
    SMS/AUC-E/KPIG/TRD) demote TRD because of its r=-0.98 collinearity with
    SMS (FLAWS §3.7); pass all 1.0 to recover the legacy equal-weight
    behaviour for backwards comparisons.
    """
    return weighted_harmonic_mean(
        [sms, auc_e, kpig, 1.0 - clamp01(trd)],
        [w_sms, w_auc_e, w_kpig, w_trd],
    )


def compute_diagnostic_ifi(
    ppl_var: float,
    bf: float,
    pc_stab: float = None,
    w_ppl_var: float = DIAG_IFI_W_PPL_VAR,
    w_bf:      float = DIAG_IFI_W_BF,
    w_pc_stab: float = DIAG_IFI_W_PC_STAB,
) -> float:
    """Diagnostic Intrinsic Fidelity Index via weighted harmonic mean.

    PPL variance, branching factor instability, and optional prompt-
    confidence stability are lower-is-better, so they are inverted before
    synthesis. PPL_var carries the largest weight (0.50) — it is the
    cleanest intra-model stability signal; BF is noisier and entropy-
    derived (0.30); PC_stab when present picks up the remaining 0.20.
    When pc_stab is None, the two-axis call uses w_ppl_var / w_bf only.
    """
    if pc_stab is None:
        return weighted_harmonic_mean(
            [1.0 - clamp01(ppl_var), 1.0 - clamp01(bf)],
            [w_ppl_var, w_bf],
        )
    return weighted_harmonic_mean(
        [
            1.0 - clamp01(ppl_var),
            1.0 - clamp01(bf),
            1.0 - clamp01(pc_stab),
        ],
        [w_ppl_var, w_bf, w_pc_stab],
    )


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
