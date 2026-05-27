import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import scores as S


def test_primary_pri_remains_weighted_linear_score():
    pri = S.compute_pri(
        consistency=0.8,
        quality=0.6,
        faithfulness=0.4,
        avg_length=S.SHORT_OUTPUT_LEN,
    )
    assert math.isclose(pri, 0.40 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4)


def test_short_output_gate_applies_once():
    pri = S.compute_pri(1.0, 1.0, 1.0, avg_length=S.SHORT_OUTPUT_LEN - 1)
    assert math.isclose(pri, S.SHORT_OUTPUT_PENALTY)


def test_diagnostic_pri_penalizes_failed_pillar():
    assert S.compute_diagnostic_pri(1.0, 0.0) == 0.0
    assert S.compute_diagnostic_pri(0.0, 1.0) == 0.0
    assert math.isclose(S.compute_diagnostic_pri(0.8, 0.8), 0.8)


def test_diagnostic_components_have_higher_is_better_polarity():
    good_ori = S.compute_diagnostic_ori(sms=0.9, auc_e=0.9, trd=0.1, kpig=0.9)
    bad_ori = S.compute_diagnostic_ori(sms=0.9, auc_e=0.9, trd=0.9, kpig=0.9)
    assert good_ori > bad_ori

    good_ifi = S.compute_diagnostic_ifi(ppl_var=0.1, bf=0.1)
    bad_ifi = S.compute_diagnostic_ifi(ppl_var=0.9, bf=0.9)
    assert good_ifi > bad_ifi


def test_dual_pillar_diagnosis_quadrants():
    assert S.compute_dual_pillar_diagnosis(0.8, 0.8) == "Robust"
    assert (
        S.compute_dual_pillar_diagnosis(0.8, 0.2)
        == "Externally Stable / Internally Fragile"
    )
    assert (
        S.compute_dual_pillar_diagnosis(0.2, 0.8)
        == "Internally Stable / Output-Sensitive"
    )
    assert S.compute_dual_pillar_diagnosis(0.2, 0.2) == "Fragile"
