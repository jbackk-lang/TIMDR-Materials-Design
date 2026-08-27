import pytest

from material_timdr.requirements import RequirementsVector, PRIMARY_FUNCTIONS
from material_timdr.figures import FIGURE_TABLE
from material_timdr.synthesis import SYNTHESIS_TABLE


def test_valid_vector_constructs():
    r = RequirementsVector(
        primary_function="conductivity",
        temperature_range_c=(-40, 120),
        environment="dry",
    )
    assert r.primary_function == "conductivity"
    assert r.as_dict()["temperature_range_c"] == (-40, 120)


def test_rejects_unknown_primary_function():
    with pytest.raises(ValueError):
        RequirementsVector(primary_function="teleportation", temperature_range_c=(0, 10))


def test_rejects_inverted_temperature_range():
    with pytest.raises(ValueError):
        RequirementsVector(primary_function="strength", temperature_range_c=(50, -10))


def test_rejects_inverted_pressure_range():
    with pytest.raises(ValueError):
        RequirementsVector(
            primary_function="strength", temperature_range_c=(0, 10),
            pressure_range_pa=(2e5, 1e5),
        )


def test_rejects_nonpositive_pressure():
    with pytest.raises(ValueError):
        RequirementsVector(
            primary_function="strength", temperature_range_c=(0, 10),
            pressure_range_pa=(-1.0, 1e5),
        )


def test_rejects_empty_environment():
    with pytest.raises(ValueError):
        RequirementsVector(primary_function="strength", temperature_range_c=(0, 10), environment="")


def test_every_primary_function_has_figure_and_synthesis_entry():
    """Zapora przed dodaniem nowej funkcji materiału bez pokrycia w
    krokach 2 i 6 procedury - patrz docstring requirements.py."""
    for fn in PRIMARY_FUNCTIONS:
        assert fn in FIGURE_TABLE, f"brak wpisu w FIGURE_TABLE dla {fn}"
        assert fn in SYNTHESIS_TABLE, f"brak wpisu w SYNTHESIS_TABLE dla {fn}"
