"""
material_timdr — TIMDR Material Design Protocol.

Implementacja 8-krokowego schematu projektowania materiału od zera przy
użyciu TIMDR (patrz README.md dla pełnego opisu i UCZCIWEGO zakresu/
ograniczeń metody). Każdy krok = osobny moduł:

1. requirements.py — wektor wymagań (RequirementsVector)
2. figures.py       — figury atomowe / hybrydyzacja (suggest_figure)
3. lattice.py + field.py — sieć atomowa + pole sygnału (Lattice, SignalField, build_signal_field)
4. spatial_timdr.py — anomalia/defekt/skręt/rezonans na grafie przestrzennym (SpatialTIMDR)
5. mapping.py       — rezonans -> funkcja materiału, z testem null-model (map_resonance_to_function)
6. synthesis.py     — heurystyki warunków syntezy (suggest_synthesis_conditions)
7. validate.py      — walidacja na zmierzonych danych (validate_against_measurements)
8. closeout.py      — checklist zamknięcia projektu (closeout_report)

pipeline.py spina kroki 1-6+8 w jedno wywołanie (design_material()) na
przykładzie syntetycznym - krok 7 (dane zmierzone) jest z natury zewnętrzny
i wywoływany osobno, patrz validate.py i tests/test_validate.py.
"""
from .requirements import RequirementsVector, PRIMARY_FUNCTIONS
from .figures import suggest_figure, FIGURE_TABLE, SP2_PLANAR, SP3_TETRAHEDRAL, SP_LINEAR
from .lattice import Lattice, honeycomb_lattice, diamond_lattice
from .field import SignalField, build_signal_field
from .steinhardt import steinhardt_q, steinhardt_field, add_steinhardt_fields
from .spatial_timdr import SpatialTIMDR, anomalia, defekt, skret, rezonans
from .mapping import MappingResult, map_resonance_to_function
from .synthesis import SynthesisSuggestion, suggest_synthesis_conditions, SYNTHESIS_TABLE
from .validate import measured_field, validate_against_measurements, ValidationResult
from .closeout import CriterionResult, CloseoutReport, closeout_report
from .pipeline import MaterialDesignResult, design_material

__all__ = [
    "RequirementsVector", "PRIMARY_FUNCTIONS",
    "suggest_figure", "FIGURE_TABLE", "SP2_PLANAR", "SP3_TETRAHEDRAL", "SP_LINEAR",
    "Lattice", "honeycomb_lattice", "diamond_lattice",
    "SignalField", "build_signal_field",
    "steinhardt_q", "steinhardt_field", "add_steinhardt_fields",
    "SpatialTIMDR", "anomalia", "defekt", "skret", "rezonans",
    "MappingResult", "map_resonance_to_function",
    "SynthesisSuggestion", "suggest_synthesis_conditions", "SYNTHESIS_TABLE",
    "measured_field", "validate_against_measurements", "ValidationResult",
    "CriterionResult", "CloseoutReport", "closeout_report",
    "MaterialDesignResult", "design_material",
]
