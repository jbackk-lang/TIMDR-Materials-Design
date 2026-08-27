import numpy as np
import pytest

from material_timdr.closeout import closeout_report
from material_timdr.mapping import map_resonance_to_function


def _fake_engine_result(n, anomaly_atoms, defekt_atoms, skret_atoms):
    return dict(
        anomaly_idx={"p1": np.array(anomaly_atoms, dtype=int)},
        defekt_idx={"p1": np.array(defekt_atoms, dtype=int)},
        skret_idx=np.array(skret_atoms, dtype=int),
        rezonans_idx=np.array([], dtype=int),
        rezonans_counts=np.zeros(n, dtype=int),
    )


def test_all_pass_when_everything_contained_and_no_critical_violations():
    n = 100
    target = np.zeros(n, dtype=bool)
    target[0:10] = True
    result = _fake_engine_result(n, anomaly_atoms=[1, 2, 3], defekt_atoms=[], skret_atoms=[])
    mapping = map_resonance_to_function(np.array([1, 2, 3]), target, n_permutations=500, seed=1)
    report = closeout_report(result, n, target, critical_region=None, mapping_result=mapping)
    assert report.criteria[0].status == "PASS"  # anomalia w target


def test_anomaly_outside_target_fails_when_over_tolerance():
    n = 100
    target = np.zeros(n, dtype=bool)
    target[0:5] = True
    result = _fake_engine_result(n, anomaly_atoms=[50, 51, 52, 53], defekt_atoms=[], skret_atoms=[])
    report = closeout_report(result, n, target, anomaly_outside_target_tolerance=0.3)
    assert report.criteria[0].status == "FAIL"


def test_empty_anomaly_passes_trivially():
    n = 50
    target = np.zeros(n, dtype=bool)
    result = _fake_engine_result(n, anomaly_atoms=[], defekt_atoms=[], skret_atoms=[])
    report = closeout_report(result, n, target)
    assert report.criteria[0].status == "PASS"
    assert report.criteria[0].value == 0.0


def test_defekt_in_critical_region_fails():
    n = 50
    target = np.zeros(n, dtype=bool)
    critical = np.zeros(n, dtype=bool)
    critical[10:20] = True
    result = _fake_engine_result(n, anomaly_atoms=[], defekt_atoms=[10, 11, 12], skret_atoms=[])
    report = closeout_report(result, n, target, critical_region=critical, defekt_in_critical_tolerance=0.1)
    assert report.criteria[1].status == "FAIL"


def test_defekt_outside_critical_region_passes():
    n = 50
    target = np.zeros(n, dtype=bool)
    critical = np.zeros(n, dtype=bool)
    critical[10:20] = True
    result = _fake_engine_result(n, anomaly_atoms=[], defekt_atoms=[30, 31], skret_atoms=[])
    report = closeout_report(result, n, target, critical_region=critical, defekt_in_critical_tolerance=0.1)
    assert report.criteria[1].status == "PASS"
    assert report.criteria[1].value == pytest.approx(0.0)


def test_missing_critical_region_marks_not_evaluated():
    n = 50
    target = np.zeros(n, dtype=bool)
    result = _fake_engine_result(n, anomaly_atoms=[], defekt_atoms=[], skret_atoms=[])
    report = closeout_report(result, n, target, critical_region=None)
    assert report.criteria[1].status == "NOT_EVALUATED"
    assert report.criteria[2].status == "NOT_EVALUATED"
    assert report.overall_status == "INCOMPLETE"


def test_missing_mapping_result_marks_not_evaluated_and_incomplete():
    n = 50
    target = np.zeros(n, dtype=bool)
    critical = np.zeros(n, dtype=bool)
    critical[0:5] = True
    result = _fake_engine_result(n, anomaly_atoms=[], defekt_atoms=[], skret_atoms=[])
    report = closeout_report(result, n, target, critical_region=critical, mapping_result=None)
    assert report.criteria[3].status == "NOT_EVALUATED"
    assert report.overall_status == "INCOMPLETE"


def test_overall_pass_requires_all_four_criteria_evaluated_and_passing():
    n = 100
    target = np.zeros(n, dtype=bool)
    target[0:10] = True
    critical = np.zeros(n, dtype=bool)
    critical[50:60] = True
    result = _fake_engine_result(n, anomaly_atoms=[1, 2], defekt_atoms=[70], skret_atoms=[71])
    mapping = map_resonance_to_function(np.array([1, 2]), target, n_permutations=500, seed=1)
    report = closeout_report(result, n, target, critical_region=critical, mapping_result=mapping)
    assert all(c.status == "PASS" for c in report.criteria)
    assert report.overall_status == "PASS"


def test_overall_fail_when_any_criterion_fails():
    n = 100
    target = np.zeros(n, dtype=bool)
    target[0:10] = True
    critical = np.zeros(n, dtype=bool)
    critical[50:60] = True
    # defekt WEWNATRZ strefy krytycznej -> to kryterium FAIL
    result = _fake_engine_result(n, anomaly_atoms=[1], defekt_atoms=[50, 51, 52, 53, 54, 55], skret_atoms=[])
    mapping = map_resonance_to_function(np.array([1]), target, n_permutations=500, seed=1)
    report = closeout_report(result, n, target, critical_region=critical, mapping_result=mapping)
    assert report.overall_status == "FAIL"


def test_summary_mentions_incomplete_warning():
    n = 50
    target = np.zeros(n, dtype=bool)
    result = _fake_engine_result(n, anomaly_atoms=[], defekt_atoms=[], skret_atoms=[])
    report = closeout_report(result, n, target)
    assert "INCOMPLETE" in report.summary_pl
    assert "nie wszystkie kryteria" in report.summary_pl
