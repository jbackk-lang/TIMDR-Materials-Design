import numpy as np
import pytest

from material_timdr.mapping import map_resonance_to_function, MappingResult


def test_perfect_overlap_gives_low_p_value():
    n = 200
    target = np.zeros(n, dtype=bool)
    target[10:20] = True  # 10 atomow
    rezonans_idx = np.arange(10, 20)  # dokladnie ta sama strefa
    r = map_resonance_to_function(rezonans_idx, target, n_permutations=2000, seed=1)
    assert r.observed_overlap == 10
    assert r.precision == pytest.approx(1.0)
    assert r.recall == pytest.approx(1.0)
    assert r.p_value < 0.01
    assert "istotnie lepsze" in r.verdict_pl


def test_disjoint_regions_give_high_p_value():
    n = 200
    target = np.zeros(n, dtype=bool)
    target[0:10] = True
    rezonans_idx = np.arange(100, 110)  # kompletnie inna strefa
    r = map_resonance_to_function(rezonans_idx, target, n_permutations=2000, seed=1)
    assert r.observed_overlap == 0
    assert r.p_value > 0.05
    assert "NIE jest statystycznie" in r.verdict_pl


def test_empty_target_region_is_not_computable():
    n = 50
    target = np.zeros(n, dtype=bool)
    rezonans_idx = np.array([1, 2, 3])
    r = map_resonance_to_function(rezonans_idx, target)
    assert r.p_value is None
    assert r.precision is None
    assert "Nie da sie policzyc" in r.verdict_pl


def test_empty_rezonans_is_not_computable():
    n = 50
    target = np.zeros(n, dtype=bool)
    target[5] = True
    r = map_resonance_to_function(np.array([], dtype=int), target)
    assert r.p_value is None


def test_reproducible_with_seed():
    n = 100
    target = np.zeros(n, dtype=bool)
    target[20:30] = True
    rez = np.arange(15, 25)
    r1 = map_resonance_to_function(rez, target, n_permutations=500, seed=42)
    r2 = map_resonance_to_function(rez, target, n_permutations=500, seed=42)
    assert r1.p_value == r2.p_value


@pytest.mark.parametrize("rez,target_slice", [
    (np.arange(0, 5), slice(0, 5)),      # perfect overlap branch
    (np.arange(50, 55), slice(0, 5)),    # disjoint branch
])
def test_scope_disclaimer_always_present_in_computable_verdict(rez, target_slice):
    n = 100
    target = np.zeros(n, dtype=bool)
    target[target_slice] = True
    r = map_resonance_to_function(rez, target, n_permutations=200, seed=1)
    assert MappingResult.SCOPE_DISCLAIMER_PL in r.verdict_pl
