import math

import pytest

from material_timdr.figures import (
    SP2_PLANAR, SP3_TETRAHEDRAL, SP_LINEAR,
    FIGURE_TABLE, suggest_figure,
    _tetrahedral_angle_deg, _planar_angle_deg,
)


def test_sp2_planar_angle_is_120():
    assert SP2_PLANAR.nominal_angle_deg == pytest.approx(120.0)
    assert SP2_PLANAR.coordination == 3
    assert SP2_PLANAR.dimensionality == "2D"


def test_sp3_tetrahedral_angle_matches_exact_geometry():
    # arccos(-1/3) w stopniach, policzone niezależnie od implementacji
    expected = math.degrees(math.acos(-1.0 / 3.0))
    assert SP3_TETRAHEDRAL.nominal_angle_deg == pytest.approx(expected, abs=1e-9)
    assert SP3_TETRAHEDRAL.nominal_angle_deg == pytest.approx(109.47122063449069, abs=1e-6)
    assert SP3_TETRAHEDRAL.coordination == 4
    assert SP3_TETRAHEDRAL.dimensionality == "3D"


def test_sp_linear_angle_is_180():
    assert SP_LINEAR.nominal_angle_deg == pytest.approx(180.0)


def test_planar_angle_formula_is_360_over_n():
    # kat MIEDZY WIAZANIAMI przy wspolnym atomie centralnym o n rowno
    # rozstawionych wiazaniach w plaszczyznie - NIE kat wewnetrzny n-kata
    # ktory tworza sami sasiedzi (to inna, latwa do pomylenia wielkosc)
    assert _planar_angle_deg(3) == pytest.approx(120.0)  # sp2 trygonalne
    assert _planar_angle_deg(6) == pytest.approx(60.0)   # 6 wiazan w plaszczyznie
    assert _planar_angle_deg(2) == pytest.approx(180.0)  # liniowe


def test_suggest_figure_conductivity_is_sp2():
    s = suggest_figure("conductivity")
    assert s.base_figure is SP2_PLANAR


def test_suggest_figure_strength_is_sp3():
    s = suggest_figure("strength")
    assert s.base_figure is SP3_TETRAHEDRAL


def test_suggest_figure_catalysis_marks_local_deviation():
    s = suggest_figure("catalysis")
    assert s.functional_zone_is_local_deviation is True


def test_suggest_figure_magnetism_has_explicit_caveat():
    s = suggest_figure("magnetism")
    assert s.caveat_pl != "", "magnetyzm powinien miec jawne zastrzezenie o slabym dopasowaniu"


def test_suggest_figure_rejects_unknown_function():
    with pytest.raises(ValueError):
        suggest_figure("nonexistent_function")


def test_figure_table_covers_all_five_functions():
    assert set(FIGURE_TABLE.keys()) == {
        "conductivity", "strength", "catalysis", "damping", "magnetism",
    }
