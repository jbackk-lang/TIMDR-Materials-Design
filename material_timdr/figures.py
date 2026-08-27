"""
figures.py — Krok 2: figury atomowe jako baza topologii.

UCZCIWOŚĆ METODOLOGICZNA (patrz README, sekcja "Zakres i ograniczenia"):
te wartości kątów/hybrydyzacji to STANDARDOWA, ustalona wiedza z chemii
strukturalnej (nie coś, co "odkrywa" TIMDR) - liczone tu wprost z geometrii,
nie wklejone z pamięci, żeby uniknąć powtórzenia błędu opisanego w skillu
timdr-signal-framework (§18 case study 6: "derive, don't just cite").

FIGURE_TABLE (funkcja materiału -> sugerowana figura atomowa) to
HEURYSTYKA Z LITERATURY MATERIAŁOZNAWCZEJ, jawnie oznaczona jako taka -
TIMDR nie ma tu żadnej roli; wchodzi do gry dopiero w polu sygnału
(field.py) i detekcji (spatial_timdr.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AtomicFigure:
    """Jedna figura atomowa - lokalna geometria wiązań wokół atomu."""

    name: str                  # "sp2_planar", "sp3_tetrahedral", "sp_linear"
    coordination: int          # liczba najbliższych sąsiadów (wiązań)
    nominal_angle_deg: float   # kąt między wiązaniami, stopnie
    dimensionality: str        # "2D" (płaszczyzna/warstwa) albo "3D" (sieć objętościowa)


def _tetrahedral_angle_deg() -> float:
    """Kąt między dwoma wiązaniami w idealnym tetraedrze - policzony z
    geometrii (kąt między wektorami od środka czworościanu do dwóch
    wierzchołków), nie wzięty z pamięci. arccos(-1/3) to standardowy wynik
    podręcznikowy, ale liczymy go tu jawnie."""
    return math.degrees(math.acos(-1.0 / 3.0))


def _planar_angle_deg(n_neighbors: int = 3) -> float:
    """Kąt między DWOMA SĄSIEDNIMI WIĄZANIAMI wychodzącymi z centralnego
    atomu o n_neighbors wiązaniach równo rozstawionych w płaszczyźnie -
    360/n_neighbors (n=3 -> sp2 trygonalne/plastry, 120°). UWAGA: to NIE
    jest kąt wewnętrzny n-kąta foremnego utworzonego przez sąsiadów (ta
    inna wielkość, 180*(n-2)/n, dla n=3 też przypadkiem daje 60° - łatwo
    pomylić te dwie różne geometrie; tu chodzi wyłącznie o kąt między
    wiązaniami przy WSPÓLNYM atomie centralnym)."""
    return 360.0 / n_neighbors


def _linear_angle_deg() -> float:
    return 180.0


SP2_PLANAR = AtomicFigure(
    name="sp2_planar",
    coordination=3,
    nominal_angle_deg=_planar_angle_deg(3),
    dimensionality="2D",
)
SP3_TETRAHEDRAL = AtomicFigure(
    name="sp3_tetrahedral",
    coordination=4,
    nominal_angle_deg=_tetrahedral_angle_deg(),
    dimensionality="3D",
)
SP_LINEAR = AtomicFigure(
    name="sp_linear",
    coordination=2,
    nominal_angle_deg=_linear_angle_deg(),
    dimensionality="1D",
)

ALL_FIGURES = {f.name: f for f in (SP2_PLANAR, SP3_TETRAHEDRAL, SP_LINEAR)}


# ---------------------------------------------------------------------
# FIGURE_TABLE — heurystyka: funkcja materiału -> sugerowana figura.
#
# Każdy wpis to (figura_bazowa, opis_lokalnej_modyfikacji). Dla katalizy i
# tłumienia funkcja NIE wynika z samej figury bazowej sieci, tylko z
# LOKALNYCH ODSTĘPSTW od niej (krawędzie, wakanse, niedopasowane domeny) -
# to jest zgodne z krokiem 4 (anomalia/skręt lokalizują te odstępstwa).
# Magnetyzm jest oznaczony jako częściowo poza zakresem tej geometrycznej
# ramy - patrz notatka niżej, nieukrywana.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSuggestion:
    base_figure: AtomicFigure
    rationale_pl: str
    functional_zone_is_local_deviation: bool
    caveat_pl: str = ""


FIGURE_TABLE: dict[str, FigureSuggestion] = {
    "conductivity": FigureSuggestion(
        base_figure=SP2_PLANAR,
        rationale_pl=(
            "Sieci sp2/2D (trójkąty, plastry) delokalizują elektrony pi w "
            "płaszczyźnie - standardowa podstawa przewodnictwa w materiałach "
            "warstwowych (grafen i pochodne)."
        ),
        functional_zone_is_local_deviation=False,
    ),
    "strength": FigureSuggestion(
        base_figure=SP3_TETRAHEDRAL,
        rationale_pl=(
            "Sieci sp3/3D (tetraedry) dają izotropowe, silne wiązania "
            "kowalencyjne we wszystkich kierunkach - standardowa podstawa "
            "twardości/wytrzymałości (sieć diamentopodobna)."
        ),
        functional_zone_is_local_deviation=False,
    ),
    "catalysis": FigureSuggestion(
        base_figure=SP2_PLANAR,
        rationale_pl=(
            "Aktywność katalityczna zwykle koncentruje się NIE w idealnej "
            "sieci bazowej, tylko w miejscach lokalnych deformacji: "
            "krawędzie, wakanse, niedokoordynowane atomy - stąd baza sp2 "
            "jako nośnik, a funkcja = NAŁOŻENIE ANOMALII i SKRĘTU w tym "
            "samym miejscu (czyli REZONANS w sensie Kroku 4 - patrz "
            "spatial_timdr.rezonans() i field.py, target_region jako "
            "strefa defektów)."
        ),
        functional_zone_is_local_deviation=True,
    ),
    "damping": FigureSuggestion(
        base_figure=SP3_TETRAHEDRAL,
        rationale_pl=(
            "Tłumienie drgań (rozpraszanie fononów) rośnie z niejednorodnością "
            "lokalnego otoczenia - baza 3D z celowo wprowadzonymi "
            "niedopasowanymi domenami/defektami rozprasza fale sieciowe "
            "skuteczniej niż idealny kryształ."
        ),
        functional_zone_is_local_deviation=True,
    ),
    "magnetism": FigureSuggestion(
        base_figure=SP3_TETRAHEDRAL,
        rationale_pl=(
            "Sam MOMENT magnetyczny pochodzi ze sparowania/niesparowania "
            "spinów elektronowych (d/f), nie z geometrii wiązań - w tym "
            "figura bazowa jest tylko nośnikiem 3D do osadzenia centrów "
            "magnetycznych. Ale STRUKTURA DOMENOWA realnego magnetyka "
            "(domeny Weissa, ściany Blocha/Néela) MA prawdziwy geometryczny "
            "odpowiednik: granica domeny to dokładnie miejsce, gdzie lokalna "
            "orientacja (tu: pole orientation_deg) skręca się między dwoma "
            "sąsiednimi obszarami - czyli funkcja = strefy SKRĘTU "
            "(spatial_timdr.skret()) między domenami, nie pojedyncza figura "
            "atomowa. To jest bezpośrednia, konkretna analogia do domain-wall "
            "w ferromagnetykach, nie luźna metafora."
        ),
        functional_zone_is_local_deviation=True,
        caveat_pl=(
            "UWAGA: to modeluje STRUKTURĘ DOMENOWĄ magnetyka (gdzie są "
            "granice domen), NIE mikroskopowe pochodzenie samego momentu "
            "magnetycznego (to wciąż fizyka spinów d/f, poza zakresem tej "
            "geometrycznej ramy) - traktuj to jako model jednego poziomu "
            "zjawiska (domeny), nie całej fizyki magnetyzmu."
        ),
    ),
}


def suggest_figure(primary_function: str) -> FigureSuggestion:
    if primary_function not in FIGURE_TABLE:
        raise ValueError(
            f"Brak wpisu w FIGURE_TABLE dla primary_function={primary_function!r}"
        )
    return FIGURE_TABLE[primary_function]
