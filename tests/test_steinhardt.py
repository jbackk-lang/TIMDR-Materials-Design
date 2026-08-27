import numpy as np
import pytest

from material_timdr.lattice import diamond_lattice, honeycomb_lattice
from material_timdr.steinhardt import steinhardt_q, steinhardt_field, add_steinhardt_fields


def _interior(lattice):
    return [i for i in range(lattice.n_atoms) if lattice.coordination(i) == lattice.figure.coordination]


def test_ideal_diamond_gives_constant_q4_q6_across_interior_atoms():
    """Nie zakladamy z literatury JAKA to wartosc - tylko ze jest STALA na
    calej idealnej sieci (kazdy atom wewnetrzny ma identyczne otoczenie z
    definicji generatora), zweryfikowane bezposrednio."""
    dm = diamond_lattice(3, 3, 3, bond_length=1.0)
    interior = _interior(dm)
    assert len(interior) >= 20
    q4_vals = [steinhardt_q(dm.positions, dm.neighbor_lists, i, 4) for i in interior]
    q6_vals = [steinhardt_q(dm.positions, dm.neighbor_lists, i, 6) for i in interior]
    assert all(v == pytest.approx(q4_vals[0], abs=1e-9) for v in q4_vals)
    assert all(v == pytest.approx(q6_vals[0], abs=1e-9) for v in q6_vals)
    # niezdegenerowane - Q4/Q6 realnie cos mierza, nie sa trywialnie 0
    assert q4_vals[0] > 0.1
    assert q6_vals[0] > 0.1


def test_q4_q6_are_rotation_invariant():
    """To jest CAŁY sens uzycia Ql zamiast surowego kata - zweryfikowane
    bezposrednio przez faktyczny obrot testowej sieci o losowa macierz
    rotacji, nie zakladane."""
    from scipy.spatial.transform import Rotation
    dm = diamond_lattice(3, 3, 3, bond_length=1.0)
    interior = _interior(dm)
    i0 = interior[0]
    q4_before = steinhardt_q(dm.positions, dm.neighbor_lists, i0, 4)
    q6_before = steinhardt_q(dm.positions, dm.neighbor_lists, i0, 6)

    R = Rotation.random(random_state=123).as_matrix()
    rotated_positions = dm.positions @ R.T
    q4_after = steinhardt_q(rotated_positions, dm.neighbor_lists, i0, 4)
    q6_after = steinhardt_q(rotated_positions, dm.neighbor_lists, i0, 6)

    assert q4_after == pytest.approx(q4_before, abs=1e-9)
    assert q6_after == pytest.approx(q6_before, abs=1e-9)


def test_isolated_atom_gives_zero():
    dm = diamond_lattice(2, 2, 2, bond_length=1.0)
    # znajdz atom z 0 sasiadow, jesli istnieje na tej malej siatce; inaczej
    # sprawdz bezposrednio logike na pustej liscie sasiadow
    assert steinhardt_q(dm.positions, [[] for _ in range(3)], 0, 4) == 0.0


def test_defect_atom_shows_different_q_value_than_ideal_neighbors():
    dm = diamond_lattice(3, 3, 3, bond_length=1.0)
    interior = _interior(dm)
    target = interior[len(interior) // 2]
    perturbed_positions = dm.positions.copy()
    perturbed_positions[target] = perturbed_positions[target] + np.array([0.3, 0.1, -0.2])

    q4_ideal = steinhardt_q(dm.positions, dm.neighbor_lists, target, 4)
    q4_defect_neighbor = steinhardt_q(perturbed_positions, dm.neighbor_lists, dm.neighbor_lists[target][0], 4)
    assert q4_defect_neighbor != pytest.approx(q4_ideal, abs=1e-6)


def test_steinhardt_field_matches_per_atom_calls():
    dm = diamond_lattice(2, 2, 2, bond_length=1.0)
    field = steinhardt_field(dm, 4)
    assert len(field) == dm.n_atoms
    for i in range(dm.n_atoms):
        assert field[i] == pytest.approx(steinhardt_q(dm.positions, dm.neighbor_lists, i, 4))


def test_add_steinhardt_fields_adds_q4_and_q6_keys():
    dm = diamond_lattice(2, 2, 2, bond_length=1.0)
    params = {}
    add_steinhardt_fields(params, dm)
    assert "q4" in params and "q6" in params
    assert len(params["q4"]) == dm.n_atoms


def test_add_steinhardt_fields_respects_custom_degrees():
    dm = diamond_lattice(2, 2, 2, bond_length=1.0)
    params = {}
    add_steinhardt_fields(params, dm, degrees=(4,))
    assert "q4" in params
    assert "q6" not in params
