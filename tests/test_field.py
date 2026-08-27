import numpy as np
import pytest

from material_timdr.lattice import honeycomb_lattice, diamond_lattice
from material_timdr.field import build_signal_field


def test_ideal_lattice_no_defects_has_zero_deviation_fields():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    f = build_signal_field(hc, seed=1)
    assert np.allclose(f.params["bond_length_dev"], 0.0)
    assert np.allclose(f.params["bond_angle_dev"], 0.0)


def test_original_lattice_object_not_mutated_by_perturbation():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    original_positions = hc.positions.copy()
    build_signal_field(hc, defect_atoms=[5, 6, 7], defect_strength=0.5, seed=1)
    assert np.array_equal(hc.positions, original_positions), (
        "build_signal_field nie moze mutowac oryginalnego obiektu Lattice"
    )


def test_defect_atoms_raise_local_bond_length_and_angle_deviation():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    target = interior[len(interior) // 2]
    f = build_signal_field(hc, defect_atoms=[target], defect_strength=0.3, seed=42)
    assert f.params["bond_length_dev"][target] > 0
    assert f.params["bond_angle_dev"][target] > 0
    # atomy daleko od defektu powinny zostac praktycznie nietkniete
    far = [i for i in interior if i != target][0]
    assert f.params["bond_length_dev"][far] == pytest.approx(0.0, abs=1e-9)


def test_defect_field_is_reproducible_with_same_seed():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    f1 = build_signal_field(hc, defect_atoms=[3, 4, 5], defect_strength=0.3, seed=7)
    f2 = build_signal_field(hc, defect_atoms=[3, 4, 5], defect_strength=0.3, seed=7)
    assert np.array_equal(f1.params["bond_length_dev"], f2.params["bond_length_dev"])


def test_different_seeds_give_different_perturbations():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    f1 = build_signal_field(hc, defect_atoms=[3, 4, 5], defect_strength=0.3, seed=1)
    f2 = build_signal_field(hc, defect_atoms=[3, 4, 5], defect_strength=0.3, seed=2)
    assert not np.array_equal(f1.params["bond_length_dev"], f2.params["bond_length_dev"])


def test_dopant_proxy_peaks_at_dopant_atom_and_decays_with_distance():
    hc = honeycomb_lattice(8, 8, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    dopant = interior[len(interior) // 2]
    f = build_signal_field(hc, dopant_atoms=[dopant], dopant_amplitude=2.0, seed=1)
    proxy = f.params["dopant_proxy"]
    assert proxy[dopant] == pytest.approx(2.0, rel=1e-6)  # exp(0) * amplitude
    # atom najdalszy od domieszki powinien miec najnizsza wartosc
    dists = np.linalg.norm(hc.positions - hc.positions[dopant], axis=1)
    farthest = int(np.argmax(dists))
    assert proxy[farthest] < proxy[dopant]


def test_target_region_defaults_to_dopant_atoms():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    f = build_signal_field(hc, dopant_atoms=[2, 9], seed=1)
    assert f.target_region[2] and f.target_region[9]
    assert f.target_region.sum() == 2


def test_target_region_can_be_specified_independently_of_dopant():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    f = build_signal_field(hc, dopant_atoms=[2], target_region_atoms=[10, 11, 12], seed=1)
    assert not f.target_region[2]
    assert all(f.target_region[i] for i in (10, 11, 12))


def test_rejects_mismatched_param_shape():
    hc = honeycomb_lattice(4, 4, bond_length=1.0)
    from material_timdr.field import SignalField
    with pytest.raises(ValueError):
        SignalField(
            lattice=hc,
            params={"bad": np.zeros(3)},
            target_region=np.zeros(hc.n_atoms, dtype=bool),
        )


def test_works_on_diamond_lattice_too():
    dm = diamond_lattice(3, 3, 3, bond_length=1.0)
    interior = [i for i in range(dm.n_atoms) if dm.coordination(i) == 4]
    target = interior[0]
    f = build_signal_field(dm, defect_atoms=[target], defect_strength=0.3, seed=1)
    assert f.params["bond_length_dev"][target] > 0
