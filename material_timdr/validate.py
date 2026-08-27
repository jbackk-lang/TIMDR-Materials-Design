"""
validate.py — Krok 7: TIMDR jako walidator KAŻDEJ wersji materiału, na
ZMIERZONYCH danych, nie tylko na modelu.

Dwie osobne funkcje:

- `measured_field()` — pakuje surowe, zmierzone tablice per-atom (naprężenia,
  proxy przewodnictwa, gęstość defektów, cokolwiek zmierzono) w SignalField,
  BEZ przechodzenia przez syntetyczną injekcję defektów z field.py
  (build_signal_field) - dane zmierzone są już "prawdziwe", nie ma czego
  symulować.
- `validate_against_measurements()` — odpala SpatialTIMDR na measured_field
  i porównuje wynikową strefę REZONANS z tą z wersji projektowej (Krok 4-5),
  przez indeks Jaccarda (|A∩B|/|A∪B|) - PROSTA miara zgodności dwóch
  URUCHOMIEŃ tego samego pipeline'u (projekt vs. pomiar), CELOWO inna od
  testu permutacyjnego z mapping.py (który sprawdza projekt vs. losowość,
  nie projekt vs. pomiar - to są dwa różne pytania, patrz README).

Zgodnie z Krokiem 7 oryginalnego schematu: jeśli `consistent=False`,
wracasz do Kroku 2 (figury) albo Kroku 6 (synteza) - `recommendation_pl`
mówi to wprost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import SignalField
from .lattice import Lattice
from .spatial_timdr import SpatialTIMDR


def measured_field(
    lattice: Lattice,
    measured_params: dict[str, np.ndarray],
    target_region: np.ndarray | None = None,
) -> SignalField:
    """Pakuje zmierzone dane w SignalField - patrz docstring modułu."""
    n = lattice.n_atoms
    if target_region is None:
        target_region = np.zeros(n, dtype=bool)
    return SignalField(
        lattice=lattice,
        params={k: np.asarray(v, float) for k, v in measured_params.items()},
        target_region=np.asarray(target_region, dtype=bool),
    )


@dataclass(frozen=True)
class ValidationResult:
    design_rezonans_idx: np.ndarray
    measured_rezonans_idx: np.ndarray
    jaccard: float
    only_in_design: np.ndarray
    only_in_measured: np.ndarray
    consistent: bool
    threshold: float
    recommendation_pl: str


def validate_against_measurements(
    design_rezonans_idx: np.ndarray,
    measured_field_obj: SignalField,
    engine: SpatialTIMDR | None = None,
    jaccard_threshold: float = 0.3,
) -> ValidationResult:
    engine = engine or SpatialTIMDR()
    measured_result = engine.analyze(measured_field_obj)
    measured_idx = measured_result["rezonans_idx"]

    design_set = set(int(i) for i in design_rezonans_idx)
    measured_set = set(int(i) for i in measured_idx)
    union = design_set | measured_set
    inter = design_set & measured_set
    jaccard = (len(inter) / len(union)) if union else 1.0  # oba puste = zgodne (brak rezonansu w obu)

    only_design = np.array(sorted(design_set - measured_set), dtype=int)
    only_measured = np.array(sorted(measured_set - design_set), dtype=int)

    consistent = jaccard >= jaccard_threshold
    if consistent:
        rec = (
            f"Zgodnosc projekt<->pomiar: Jaccard={jaccard:.2f} >= "
            f"prog={jaccard_threshold} - strefy rezonansu z projektu i z "
            f"pomiaru pokrywaja sie wystarczajaco. Mozna zamknac projekt "
            f"(Krok 8)."
        )
    else:
        rec = (
            f"Zgodnosc projekt<->pomiar: Jaccard={jaccard:.2f} < "
            f"prog={jaccard_threshold} - strefy rezonansu z pomiaru NIE "
            f"pasuja do zalozen projektowych. Wroc do Kroku 2 (figury "
            f"atomowe) jesli problem jest strukturalny (zla geometria "
            f"bazowa), albo do Kroku 6 (synteza) jesli geometria jest "
            f"dobra ale proces jej nie realizuje."
        )

    return ValidationResult(
        design_rezonans_idx=np.asarray(design_rezonans_idx, dtype=int),
        measured_rezonans_idx=measured_idx,
        jaccard=jaccard,
        only_in_design=only_design,
        only_in_measured=only_measured,
        consistent=consistent,
        threshold=jaccard_threshold,
        recommendation_pl=rec,
    )
