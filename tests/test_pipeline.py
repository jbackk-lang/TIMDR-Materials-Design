import numpy as np
import pytest

from material_timdr import RequirementsVector, design_material
from material_timdr.lattice import honeycomb_lattice, diamond_lattice


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


def test_widen_target_to_dopant_neighbors_default_false_matches_old_behavior():
    """Domyslnie (bez flagi) target_region powinien byc DOKLADNIE dopant_atoms,
    tak jak przed dodaniem tej opcji - zadnej cichej zmiany zachowania."""
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    result = design_material(req, lattice_size=(6, 6), dopant_atoms=[37], seed=7)
    assert set(int(i) for i in np.where(result.field.target_region)[0]) == {37}


def test_widen_target_to_dopant_neighbors_true_includes_neighbors():
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    result = design_material(
        req, lattice_size=(6, 6), dopant_atoms=[37], dopant_amplitude=2.5,
        widen_target_to_dopant_neighbors=True, seed=7,
    )
    expected = {37} | set(int(i) for i in hc.neighbor_lists[37])
    actual = set(int(i) for i in np.where(result.field.target_region)[0])
    assert actual == expected
    assert len(expected) > 1  # sanity - faktycznie poszerzone, nie zdegenerowane


def test_widen_target_to_dopant_neighbors_ignored_if_target_region_explicit():
    """Flaga NIE powinna nadpisywac jawnie podanego target_region_atoms -
    'widen' dziala tylko jako fallback, gdy uzytkownik niczego nie wybral."""
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    result = design_material(
        req, lattice_size=(6, 6), dopant_atoms=[37],
        target_region_atoms=[37],  # jawnie wskazane, wąskie
        widen_target_to_dopant_neighbors=True, seed=7,
    )
    assert set(int(i) for i in np.where(result.field.target_region)[0]) == {37}


def test_dopant_sigma_is_actually_passed_through_to_field():
    """Przed dodaniem tego parametru design_material() w ogole nie
    wystawialo dopant_sigma - zawsze uzywany byl domyslny sigma=2*bond_length
    niezaleznie od tego co by ktos podal. Ten test lapie regresje, gdyby
    parametr zostal po cichu zignorowany (por. bond_length_dev daleko od
    domieszki - wieksze sigma = wolniej zanikajacy dopant_proxy)."""
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    narrow = design_material(req, lattice_size=(8, 8), dopant_atoms=[40], dopant_sigma=0.3, seed=1)
    wide = design_material(req, lattice_size=(8, 8), dopant_atoms=[40], dopant_sigma=5.0, seed=1)
    # ten sam atom daleko od domieszki - z waskim sigma pole powinno tam
    # byc bliskie zeru, z szerokim - wyraznie wieksze
    far_atom = 0
    assert narrow.field.params["dopant_proxy"][far_atom] < wide.field.params["dopant_proxy"][far_atom]


def test_narrow_dopant_sigma_with_widened_target_can_reach_pass():
    """Uczciwy test pozytywny: z wystarczajaco waskim sigma i poszerzona
    strefa docelowa (dopant + sasiedzi) PASS jest OSIAGALNY, nie tylko
    FAIL/INCOMPLETE - pipeline nie jest strukturalnie skazany na
    niepowodzenie, tylko wymaga sensownych parametrow (patrz README
    'Zakres i ograniczenia')."""
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    hc = honeycomb_lattice(10, 10, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    dopant = interior[len(interior) // 2]
    result = design_material(
        req, lattice_size=(10, 10),
        dopant_atoms=[dopant], dopant_amplitude=1.0, dopant_sigma=0.6,
        widen_target_to_dopant_neighbors=True,
        critical_region_atoms=interior[:4],
        seed=3,
    )
    assert result.closeout.overall_status == "PASS", result.closeout.summary_pl


def test_diamond_dopant_scenario_can_reach_pass_after_boundary_condition_fix():
    """Przed poprawka warunku brzegowego (Lattice.bulk_mask(),
    SpatialTIMDR.BOUNDARY_SENSITIVE_FIELDS) sieci 3D (diament/krzem/german)
    byly STRUKTURALNIE skazane na FAIL niezaleznie od dopant_amplitude/sigma -
    fale Q4/Q6 na atomach brzegowych skonczonej sieci zalewaly wynik.
    Ten test dowodzi, ze PASS jest teraz realnie osiagalny rowniez dla
    sieci 3D, nie tylko 2D (patrz test_narrow_dopant_sigma_with_widened_target_can_reach_pass
    dla analogicznego testu na honeycomb)."""
    dia = diamond_lattice(6, 6, 3, bond_length=1.54)
    bulk = np.where(dia.bulk_mask())[0].tolist()
    assert len(bulk) >= 4, "siatka za mala do sensownego testu"
    dopant = bulk[len(bulk) // 2]
    critical = bulk[:4]

    req = RequirementsVector(primary_function="strength", temperature_range_c=(0, 500))
    result = design_material(
        req, lattice_size=(6, 6, 3), bond_length=1.54,
        dopant_atoms=[dopant], dopant_amplitude=1.0, dopant_sigma=0.5 * 1.54,
        widen_target_to_dopant_neighbors=True,
        critical_region_atoms=critical,
        n_permutations=1000, seed=1,
    )
    assert result.closeout.overall_status == "PASS", result.closeout.summary_pl


def test_widen_target_to_dopant_neighbors_noop_without_dopant():
    """Bez dopant_atoms flaga nie ma czego poszerzac - target_region
    powinien zostac pusty, tak jak bez flagi."""
    req = RequirementsVector(primary_function="conductivity", temperature_range_c=(0, 50))
    result = design_material(
        req, lattice_size=(6, 6), widen_target_to_dopant_neighbors=True, seed=1,
    )
    assert result.field.target_region.sum() == 0
