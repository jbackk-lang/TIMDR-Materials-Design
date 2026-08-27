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
    dopant_sigma: float | None = None,
    target_region_atoms: list[int] | None = None,
    critical_region_atoms: list[int] | None = None,
    bond_length: float = 1.0,
    engine: SpatialTIMDR | None = None,
    n_permutations: int = 2000,
    seed: int | None = None,
    widen_target_to_dopant_neighbors: bool = False,
) -> MaterialDesignResult:
    """
    widen_target_to_dopant_neighbors: gdy True I `target_region_atoms` nie
    zostal podany jawnie I sa dopant_atoms, strefa docelowa to
    dopant_atoms + ich BEZPOSREDNI SASIEDZI w sieci, nie sam atom
    domieszki. Domyslnie False (bez zmiany zachowania wzgledem
    poprzednich wersji - dawny default field.py: target_region ==
    dopant_atoms).

    dopant_sigma: przekazywane wprost do build_signal_field() (Krok 3) -
    przedtem design_material() w ogole nie wystawialo tego parametru,
    wiec zawsze uzywany byl domyslny sigma=2*bond_length niezaleznie od
    tego, co ktos by chcial. Mniejsze sigma => waskszy gaussowski
    "pagorek" domieszki => mniejsza szansa, ze anomalia (Krok 4) rozleje
    sie poza target_region (Krok 8, kryterium 1) - to jest realny lever
    do uzyskania PASS, nie tylko widen_target_to_dopant_neighbors.

    UZASADNIENIE (ta sama lekcja co w examples/demo_graphene_dopant.py,
    tu podniesiona do prawdziwej, testowanej opcji biblioteki zamiast
    kodu wklejonego ręcznie w skrypcie demo): `dopant_proxy` to gladki
    gaussowski "pagorek" (patrz field.py) - jego DYSKRETNY GRADIENT (na
    ktorym opiera sie defekt()) jest ZEROWY dokladnie w SZCZYCIE pagorka
    (lokalne maksimum, symetryczne roznice sie znosza) i najwiekszy na
    "zboczach" wokol niego. Z target_region=[sam_atom_domieszki] Krok 5
    (mapping.py) prawie zawsze dostaje p=1.0 (0% pokrycia) NIE dlatego,
    ze pipeline jest zle skonfigurowany, tylko dlatego, ze rezonans
    faktycznie tworzy PIERScIEN wokol szczytu, nie sam szczyt - to jest
    uczciwa, sprawdzalna wlasciwosc geometrii gradientu (zweryfikowana
    bezposrednio w tescie ponizej), nie blad kodu. Ta opcja NIE gwarantuje
    PASS w Kroku 8 - kryterium 'anomalia tylko w strefie docelowej' wciaz
    moze legalnie FAIL, jesli dopant_amplitude/sigma sa duze wzgledem
    rozmiaru strefy (patrz [Interpretacja] w demo_graphene_dopant.py).
    """
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

    effective_target = target_region_atoms
    if widen_target_to_dopant_neighbors and target_region_atoms is None and dopant_atoms:
        widened: set[int] = set()
        for d in dopant_atoms:
            widened.add(d)
            widened.update(lattice.neighbor_lists[d])
        effective_target = sorted(widened)

    # Krok 3: pole sygnału
    field = build_signal_field(
        lattice,
        defect_atoms=defect_atoms,
        defect_strength=defect_strength,
        dopant_atoms=dopant_atoms,
        dopant_amplitude=dopant_amplitude,
        dopant_sigma=dopant_sigma,
        target_region_atoms=effective_target,
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
