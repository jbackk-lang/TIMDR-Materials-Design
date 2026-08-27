"""
pipeline.py — spina Kroki 1-6 + 8 w jedno wywołanie na przykładzie
SYNTETYCZNYM (Krok 7, walidacja na zmierzonych danych, jest z natury
zewnętrzny i wywoływany osobno - patrz validate.py i tests/test_validate.py,
oraz examples/demo_graphene_dopant.py po pełny przykład użycia razem z
Krokiem 7).

`design_material()` NIE jest "magicznym przyciskiem" - buduje sieć i pole
sygnału na podstawie PARAMETRÓW, KTÓRE TY PODAJESZ (rozmiar sieci, gdzie są
defekty/domieszki, gdzie jest target_region) - automatyzuje tylko
PRZEPŁYW między krokami (Krok 2 -> Krok 3 -> Krok 4 -> Krok 5 -> Krok 6 ->
Krok 8), nie decyzje projektowe, które w oryginalnym schemacie należą do
inżyniera."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .requirements import RequirementsVector
from .figures import suggest_figure, FigureSuggestion
from .lattice import Lattice, honeycomb_lattice, diamond_lattice
from .field import SignalField, build_signal_field
from .spatial_timdr import SpatialTIMDR
from .mapping import MappingResult, map_resonance_to_function
from .synthesis import SynthesisSuggestion, suggest_synthesis_conditions
from .closeout import CloseoutReport, closeout_report


@dataclass(frozen=True)
class MaterialDesignResult:
    requirements: RequirementsVector
    figure_suggestion: FigureSuggestion
    synthesis_suggestion: SynthesisSuggestion
    field: SignalField
    engine_result: dict
    mapping_result: MappingResult
    closeout: CloseoutReport


def design_material(
    requirements: RequirementsVector,
    lattice_size: tuple[int, int] | tuple[int, int, int] = (8, 8),
    defect_atoms: list[int] | None = None,
    defect_strength: float = 0.3,
    dopant_atoms: list[int] | None = None,
    dopant_amplitude: float = 1.0,
    target_region_atoms: list[int] | None = None,
    critical_region_atoms: list[int] | None = None,
    bond_length: float = 1.0,
    engine: SpatialTIMDR | None = None,
    n_permutations: int = 2000,
    seed: int | None = None,
) -> MaterialDesignResult:
    # Krok 2: figura atomowa sugerowana dla tej funkcji
    figure_suggestion = suggest_figure(requirements.primary_function)

    # Krok 2/3: sieć bazowa - sp2 -> honeycomb (2D), sp3 -> diamond (3D)
    if figure_suggestion.base_figure.dimensionality == "2D":
        if len(lattice_size) != 2:
            raise ValueError("Figura 2D (sp2) wymaga lattice_size=(n1, n2)")
        lattice = honeycomb_lattice(*lattice_size, bond_length=bond_length)
    else:
        if len(lattice_size) != 3:
            raise ValueError("Figura 3D (sp3) wymaga lattice_size=(n1, n2, n3)")
        lattice = diamond_lattice(*lattice_size, bond_length=bond_length)

    # Krok 3: pole sygnału
    field = build_signal_field(
        lattice,
        defect_atoms=defect_atoms,
        defect_strength=defect_strength,
        dopant_atoms=dopant_atoms,
        dopant_amplitude=dopant_amplitude,
        target_region_atoms=target_region_atoms,
        seed=seed,
    )

    # Krok 4: TIMDR na polu materiału
    engine = engine or SpatialTIMDR()
    engine_result = engine.analyze(field)

    # Krok 5: mapowanie rezonansu na funkcję
    mapping_result = map_resonance_to_function(
        engine_result["rezonans_idx"], field.target_region,
        n_permutations=n_permutations, seed=seed,
    )

    # Krok 6: sugestia syntezy (informacyjna, patrz synthesis.py)
    synthesis_suggestion = suggest_synthesis_conditions(requirements.primary_function)

    # Krok 8: checklist zamknięcia
    critical_region = None
    if critical_region_atoms is not None:
        critical_region = np.zeros(lattice.n_atoms, dtype=bool)
        for i in critical_region_atoms:
            critical_region[i] = True

    closeout = closeout_report(
        engine_result, lattice.n_atoms, field.target_region,
        critical_region=critical_region, mapping_result=mapping_result,
    )

    return MaterialDesignResult(
        requirements=requirements,
        figure_suggestion=figure_suggestion,
        synthesis_suggestion=synthesis_suggestion,
        field=field,
        engine_result=engine_result,
        mapping_result=mapping_result,
        closeout=closeout,
    )
