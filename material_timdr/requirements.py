"""
requirements.py — Krok 1: wektor wymagań materiału.

Zamierzone: to jest CZYSTA STRUKTURA DANYCH, nie coś, co TIMDR "oblicza".
Punkt wejścia całej procedury - użytkownik/inżynier decyduje, co materiał
ma robić; TIMDR wchodzi do gry dopiero od kroku 3 (pole sygnału).

Zestaw `PRIMARY_FUNCTIONS` jest celowo mały i zamknięty (enum-jak string) -
każda wartość musi mieć odpowiadający wpis w figures.FIGURE_TABLE i
synthesis.SYNTHESIS_TABLE, sprawdzane testem
`test_requirements.py::test_every_primary_function_has_figure_and_synthesis_entry`,
żeby nie dało się po cichu dodać funkcji materiału bez pokrycia w dalszych
krokach procedury.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRIMARY_FUNCTIONS = (
    "conductivity",  # przewodzenie (elektronowe)
    "strength",      # wytrzymałość mechaniczna / twardość
    "catalysis",     # kataliza (aktywność powierzchniowa)
    "damping",       # tłumienie drgań (rozpraszanie fononów)
    "magnetism",     # magnetyzm (lokalne, niesparowane spiny)
)


@dataclass(frozen=True)
class RequirementsVector:
    """Krok 1 procedury - wektor wymagań, NIE opis słowny.

    primary_function: jedna z PRIMARY_FUNCTIONS - co materiał ma robić.
    temperature_range_c: (min, max) temperatura pracy w stopniach C.
    pressure_range_pa: (min, max) ciśnienie pracy w paskalach.
    environment: krótki tag środowiska - "dry", "humid", "chemically_aggressive",
        "vacuum", "inert_gas" - słownik otwarty celowo (środowisk pracy jest
        więcej niż da się zamknąć w enum), ale walidowany minimalnie (niepusty
        string) w __post_init__.
    """

    primary_function: str
    temperature_range_c: tuple[float, float]
    pressure_range_pa: tuple[float, float] = (101325.0, 101325.0)
    environment: str = "dry"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.primary_function not in PRIMARY_FUNCTIONS:
            raise ValueError(
                f"primary_function={self.primary_function!r} spoza domeny "
                f"PRIMARY_FUNCTIONS={PRIMARY_FUNCTIONS}"
            )
        lo, hi = self.temperature_range_c
        if lo > hi:
            raise ValueError(f"temperature_range_c: min ({lo}) > max ({hi})")
        plo, phi = self.pressure_range_pa
        if plo > phi:
            raise ValueError(f"pressure_range_pa: min ({plo}) > max ({phi})")
        if plo <= 0:
            raise ValueError(f"pressure_range_pa: wartości muszą być > 0, dostano {plo}")
        if not self.environment:
            raise ValueError("environment nie może być puste")

    def as_dict(self) -> dict:
        return dict(
            primary_function=self.primary_function,
            temperature_range_c=self.temperature_range_c,
            pressure_range_pa=self.pressure_range_pa,
            environment=self.environment,
            notes=self.notes,
        )
