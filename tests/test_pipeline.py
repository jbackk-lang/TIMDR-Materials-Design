import numpy as np
import pytest

from material_timdr import RequirementsVector, design_material
from material_timdr.lattice import honeycomb_lattice


def test_conductivity_pipeline_end_to_end_runs_without_error():
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(-20, 80))
    interior_hint = list(range(20, 30))  # przyblizone atomy "wewnetrzne" 8x8 honeycomb
    result = design_material(
        req, lattice_size=(8, 8),
        dopant_atoms=[interior_hint[0]], dopant_amplitude=2.0,
        target_region_atoms=[interior_hint[0]],
        seed=1,
    )
    assert result.figure_suggestion.base_figure.name == "sp2_planar"
    assert result.field.lattice.n_atoms == 128
    assert result.mapping_result is not None
    assert result.closeout.overall_status in ("PASS", "FAIL", "INCOMPLETE")


def test_strength_pipeline_uses_diamond_3d_lattice():
    req = RequirementsVector(primary_function="strength", temperature_range_c=(0, 500))
    result = design_material(req, lattice_size=(3, 3, 3), seed=1)
    assert result.figure_suggestion.base_figure.name == "sp3_tetrahedral"
    assert result.field.lattice.positions.shape[1] == 3
    assert "orientation_deg" not in result.field.params  # 3D - brak skretu
    assert "q4" in result.field.params and "q6" in result.field.params  # ale ma Q4/Q6


def test_wrong_dimensionality_lattice_size_raises():
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 10))
    with pytest.raises(ValueError):
        design_material(req, lattice_size=(3, 3, 3))  # sp2 potrzebuje 2D rozmiaru


def test_catalysis_pipeline_defect_zone_becomes_target_region():
    req = RequirementsVector(primary_function="catalysis", temperature_range_c=(20, 300))
    hc = honeycomb_lattice(8, 8, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    defect_zone = interior[:3]
    result = design_material(
        req, lattice_size=(8, 8),
        defect_atoms=defect_zone, defect_strength=0.4,
        target_region_atoms=defect_zone,
        seed=3,
    )
    assert result.field.target_region.sum() == 3
    assert result.synthesis_suggestion.notes_pl


def test_result_is_reproducible_with_same_seed():
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    r1 = design_material(req, lattice_size=(6, 6), dopant_atoms=[5], seed=42)
    r2 = design_material(req, lattice_size=(6, 6), dopant_atoms=[5], seed=42)
    assert np.array_equal(r1.field.params["dopant_proxy"], r2.field.params["dopant_proxy"])
    assert r1.mapping_result.p_value == r2.mapping_result.p_value
