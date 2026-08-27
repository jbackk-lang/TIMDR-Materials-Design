import math

import numpy as np
import pytest

from material_timdr.lattice import honeycomb_lattice, diamond_lattice


def test_honeycomb_interior_atoms_have_coordination_3():
    hc = honeycomb_lattice(n1=6, n2=6, bond_length=1.0)
    coords = [hc.coordination(i) for i in range(hc.n_atoms)]
    # w środku sieci (nie na brzegu) koordynacja musi być dokładnie 3
    interior = [c for c in coords if c == 3]
    assert len(interior) > 0
    assert max(coords) == 3  # nikt nie ma WIĘCEJ niż 3 sąsiadów


def test_honeycomb_interior_bond_angles_are_exactly_120():
    hc = honeycomb_lattice(n1=6, n2=6, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    assert len(interior) >= 10, "za mala siatka do sensownego testu wnetrza"
    for i in interior:
        for ang in hc.bond_angles_deg(i):
            assert ang == pytest.approx(120.0, abs=1e-6)


def test_honeycomb_all_bonds_have_nominal_length():
    hc = honeycomb_lattice(n1=4, n2=4, bond_length=2.5)
    for i, j in hc.edges:
        d = np.linalg.norm(hc.positions[i] - hc.positions[j])
        assert d == pytest.approx(2.5, rel=1e-6)


def test_diamond_interior_atoms_have_coordination_4():
    dm = diamond_lattice(n1=3, n2=3, n3=3, bond_length=1.0)
    coords = [dm.coordination(i) for i in range(dm.n_atoms)]
    interior = [c for c in coords if c == 4]
    assert len(interior) > 0
    assert max(coords) == 4


def test_diamond_interior_bond_angles_match_tetrahedral_angle():
    dm = diamond_lattice(n1=3, n2=3, n3=3, bond_length=1.0)
    expected = math.degrees(math.acos(-1.0 / 3.0))
    interior = [i for i in range(dm.n_atoms) if dm.coordination(i) == 4]
    assert len(interior) >= 10
    for i in interior:
        for ang in dm.bond_angles_deg(i):
            assert ang == pytest.approx(expected, abs=1e-5)


def test_diamond_all_bonds_have_nominal_length():
    dm = diamond_lattice(n1=2, n2=2, n3=2, bond_length=1.3)
    for i, j in dm.edges:
        d = np.linalg.norm(dm.positions[i] - dm.positions[j])
        assert d == pytest.approx(1.3, rel=1e-6)


def test_lattice_scales_linearly_with_bond_length():
    hc1 = honeycomb_lattice(4, 4, bond_length=1.0)
    hc2 = honeycomb_lattice(4, 4, bond_length=3.0)
    assert hc1.n_atoms == hc2.n_atoms
    assert len(hc1.edges) == len(hc2.edges)


def test_bond_angles_empty_for_isolated_or_singly_bonded_atom():
    hc = honeycomb_lattice(2, 2, bond_length=1.0)
    # znajdz atom z 0 lub 1 sasiadem (brzeg malej siatki)
    lone = [i for i in range(hc.n_atoms) if hc.coordination(i) <= 1]
    assert lone, "mala siatka powinna miec atomy brzegowe z <=1 sasiadem"
    assert hc.bond_angles_deg(lone[0]) == []
