import numpy as np
import pytest

from material_timdr.lattice import honeycomb_lattice
from material_timdr.field import build_signal_field
from material_timdr.spatial_timdr import SpatialTIMDR
from material_timdr.validate import measured_field, validate_against_measurements
from material_timdr.synthesis import suggest_synthesis_conditions, SYNTHESIS_TABLE


def test_synthesis_table_covers_all_functions():
    for fn in ("conductivity", "strength", "catalysis", "damping", "magnetism"):
        s = suggest_synthesis_conditions(fn)
        assert s.notes_pl


def test_synthesis_rejects_unknown_function():
    with pytest.raises(ValueError):
        suggest_synthesis_conditions("nope")


def test_magnetism_synthesis_flags_weak_grounding():
    s = suggest_synthesis_conditions("magnetism")
    assert "NAJSLABIEJ" in s.notes_pl.upper() or "SŁABIEJ" in s.notes_pl or "slabiej" in s.notes_pl.lower()


def test_measured_field_wraps_raw_arrays_without_defect_injection():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    n = hc.n_atoms
    raw = {"conductivity_proxy": np.random.default_rng(1).normal(size=n)}
    f = measured_field(hc, raw)
    assert np.array_equal(f.params["conductivity_proxy"], raw["conductivity_proxy"])
    assert f.target_region.sum() == 0  # domyslnie brak target_region


def test_identical_field_used_for_design_and_measurement_is_perfectly_consistent():
    hc = honeycomb_lattice(8, 8, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    f = build_signal_field(hc, defect_atoms=interior[:3], defect_strength=0.4, seed=3)
    engine = SpatialTIMDR(rezonans_min=1)
    design_result = engine.analyze(f)

    # "pomiar" to DOKLADNIE te same dane co projekt - powinno dac Jaccard=1
    measured = measured_field(hc, f.params)
    v = validate_against_measurements(design_result["rezonans_idx"], measured, engine=engine)
    assert v.jaccard == pytest.approx(1.0)
    assert v.consistent is True
    assert len(v.only_in_design) == 0
    assert len(v.only_in_measured) == 0


def test_completely_different_measurement_is_flagged_inconsistent():
    hc = honeycomb_lattice(8, 8, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    f_design = build_signal_field(hc, defect_atoms=[interior[0]], defect_strength=0.5, seed=1)
    engine = SpatialTIMDR(rezonans_min=1)
    design_result = engine.analyze(f_design)

    # pomiar zupelnie inny obszar defektu
    f_measured_source = build_signal_field(hc, defect_atoms=[interior[-1]], defect_strength=0.5, seed=99)
    measured = measured_field(hc, f_measured_source.params)
    v = validate_against_measurements(design_result["rezonans_idx"], measured, engine=engine, jaccard_threshold=0.3)
    assert v.consistent is False
    assert "Wroc do Kroku" in v.recommendation_pl


def test_both_empty_rezonans_counts_as_consistent():
    hc = honeycomb_lattice(4, 4, bond_length=1.0)
    n = hc.n_atoms
    measured = measured_field(hc, {"flat": np.zeros(n)})
    engine = SpatialTIMDR(rezonans_min=5)  # nieosiagalny prog -> puste rezonans
    v = validate_against_measurements(np.array([], dtype=int), measured, engine=engine)
    assert v.jaccard == pytest.approx(1.0)
    assert v.consistent is True
