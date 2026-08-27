"""
synthesis.py — Krok 6: sugestie warunków syntezy pod figury, nie pod skład.

UCZCIWOŚĆ METODOLOGICZNA (patrz README): to jest TABELA HEURYSTYK Z
LITERATURY MATERIAŁOZNAWCZEJ, jakościowa i celowo bez fałszywej precyzji
(żadnych konkretnych stopni/paskali "z powietrza") - TIMDR nie ma tu ŻADNEJ
roli obliczeniowej. Funkcja tego modułu jest wyłącznie informacyjna: łączy
sugerowaną figurę atomową (figures.py, Krok 2) z ogólnym kierunkiem warunków
syntezy, które W LITERATURZE sprzyjają tej hybrydyzacji. Realny dobór
parametrów syntezy wymaga eksperymentu/literatury specyficznej dla
konkretnego układu chemicznego, nie tej tabeli.
"""
from __future__ import annotations

from dataclasses import dataclass

from .figures import SP2_PLANAR, SP3_TETRAHEDRAL


@dataclass(frozen=True)
class SynthesisSuggestion:
    favored_temperature_pl: str
    favored_cooling_pl: str
    favored_pressure_pl: str
    notes_pl: str


SYNTHESIS_TABLE: dict[str, SynthesisSuggestion] = {
    "conductivity": SynthesisSuggestion(
        favored_temperature_pl="wysoka (sprzyja katalitycznemu wzrostowi warstwowemu sp2, np. CVD)",
        favored_cooling_pl="kontrolowane, umiarkowane - zbyt szybkie chłodzenie sprzyja defektom punktowym psującym delokalizację",
        favored_pressure_pl="niskie/atmosferyczne, typowe dla wzrostu warstwowego",
        notes_pl="Kierunek zgodny z syntezą materiałów warstwowych sp2 (grafen i pochodne) - nie parametry konkretnego reaktora.",
    ),
    "strength": SynthesisSuggestion(
        favored_temperature_pl="bardzo wysoka",
        favored_cooling_pl="powolne, kontrolowane - szybkie chłodzenie zamraża naprężenia sieciowe",
        favored_pressure_pl="wysokie (sprzyja gęstemu upakowaniu sp3/3D)",
        notes_pl="Kierunek zgodny z syntezą sieci diamentopodobnych (wysokie P/T lub CVD niskotlenowe) - jakościowo, nie ilościowo.",
    ),
    "catalysis": SynthesisSuggestion(
        favored_temperature_pl="umiarkowana, z etapem kontrolowanego trawienia/aktywacji powierzchni",
        favored_cooling_pl="szybsze niż dla 'strength' - celowo WIĘCEJ defektów krawędziowych/wakancji",
        favored_pressure_pl="zależne od układu - zwykle niższe niż dla gęstych sieci 3D",
        notes_pl="Funkcja pochodzi z LOKALNYCH odstępstw od sieci bazowej (patrz figures.FIGURE_TABLE) - synteza celowo NIE dąży do idealnego kryształu.",
    ),
    "damping": SynthesisSuggestion(
        favored_temperature_pl="umiarkowana, z domieszkowaniem wprowadzającym niejednorodność",
        favored_cooling_pl="szybkie/nierównomierne - sprzyja niedopasowanym domenom rozpraszającym fonony",
        favored_pressure_pl="zależne od układu",
        notes_pl="Podobnie jak przy katalizie: funkcja pochodzi z kontrolowanej niejednorodności, nie z idealnej sieci.",
    ),
    "magnetism": SynthesisSuggestion(
        favored_temperature_pl="zależne od konkretnego centrum magnetycznego (d/f) - brak ogólnej reguły",
        favored_cooling_pl="zależne od układu",
        favored_pressure_pl="zależne od układu",
        notes_pl=(
            "NAJSŁABIEJ ugruntowany wpis w tej tabeli, spójnie z zastrzeżeniem "
            "w figures.FIGURE_TABLE['magnetism'].caveat_pl - magnetyzm zależy "
            "głównie od chemii centrów d/f, nie od geometrii sp2/sp3, więc ten "
            "wpis ma niską wartość informacyjną i jest tu tylko dla kompletności "
            "tabeli (patrz test_requirements.py::test_every_primary_function_has_figure_and_synthesis_entry)."
        ),
    ),
}


def suggest_synthesis_conditions(primary_function: str) -> SynthesisSuggestion:
    if primary_function not in SYNTHESIS_TABLE:
        raise ValueError(
            f"Brak wpisu w SYNTHESIS_TABLE dla primary_function={primary_function!r}"
        )
    return SYNTHESIS_TABLE[primary_function]
