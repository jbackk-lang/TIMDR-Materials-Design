"""
closeout.py — Krok 8: zamknięcie projektu (materiał = stabilne figury +
kontrolowane rezonanse).

Cztery kryteria wprost z oryginalnego schematu, każde jako osobne,
mierzalne sprawdzenie (nie tylko "wygląda dobrze"):

1. ANOMALIA tylko tam, gdzie potrzebna (np. kataliza w target_region) -
   sprawdzane jako: jaki ułamek WSZYSTKICH wskazań anomalia leży POZA
   target_region.
2. DEFEKT nie niszczy ciągłości tam, gdzie potrzebna wytrzymałość -
   sprawdzane jako: jaki ułamek strefy critical_region (jeśli podana) jest
   dotknięty przez defekt.
3. SKRĘT nie psuje orientacji domen krytycznych - analogicznie do (2), na
   skret_idx.
4. REZONANS jest tam, gdzie chcemy funkcji, nie gdzie się sypie - brane
   WPROST z mapping.MappingResult (Krok 5), nie liczone od nowa tutaj.

Każde kryterium ma status PASS/FAIL/NOT_EVALUATED (to ostatnie gdy brakuje
danych do sprawdzenia - np. nie podano critical_region) - overall_status
jest PASS tylko gdy WSZYSTKIE ocenione kryteria są PASS i ŻADNE nie jest
NOT_EVALUATED; w przeciwnym razie INCOMPLETE (brakuje danych) albo FAIL.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .mapping import MappingResult


@dataclass(frozen=True)
class CriterionResult:
    name_pl: str
    status: str  # "PASS" | "FAIL" | "NOT_EVALUATED"
    value: float | None
    tolerance: float | None
    detail_pl: str


@dataclass(frozen=True)
class CloseoutReport:
    criteria: list[CriterionResult]
    overall_status: str  # "PASS" | "FAIL" | "INCOMPLETE"
    summary_pl: str


def _frac(numerator_set: set[int], denominator_set: set[int]) -> float | None:
    if not denominator_set:
        return None
    return len(numerator_set & denominator_set) / len(denominator_set)


def closeout_report(
    engine_result: dict,
    n_atoms: int,
    target_region: np.ndarray,
    critical_region: np.ndarray | None = None,
    mapping_result: MappingResult | None = None,
    anomaly_outside_target_tolerance: float = 0.5,
    defekt_in_critical_tolerance: float = 0.1,
    skret_in_critical_tolerance: float = 0.1,
) -> CloseoutReport:
    target_set = set(int(i) for i in np.where(target_region)[0])
    critical_set = set(int(i) for i in np.where(critical_region)[0]) if critical_region is not None else set()

    # --- Kryterium 1: ANOMALIA tylko tam, gdzie potrzebna ---
    all_anomaly: set[int] = set()
    for idxs in engine_result.get("anomaly_idx", {}).values():
        all_anomaly |= set(int(i) for i in idxs)
    if not all_anomaly:
        c1 = CriterionResult(
            "Anomalia tylko w strefie docelowej", "PASS", 0.0, anomaly_outside_target_tolerance,
            "Brak jakichkolwiek wskazan anomalia - kryterium spelnione trywialnie.",
        )
    else:
        outside_frac = len(all_anomaly - target_set) / len(all_anomaly)
        status = "PASS" if outside_frac <= anomaly_outside_target_tolerance else "FAIL"
        c1 = CriterionResult(
            "Anomalia tylko w strefie docelowej", status, outside_frac, anomaly_outside_target_tolerance,
            f"{outside_frac:.2f} wskazan anomalia lezy POZA target_region (tolerancja {anomaly_outside_target_tolerance}).",
        )

    # --- Kryterium 2: DEFEKT nie niszczy critical_region ---
    all_defekt: set[int] = set()
    for idxs in engine_result.get("defekt_idx", {}).values():
        all_defekt |= set(int(i) for i in idxs)
    if critical_region is None:
        c2 = CriterionResult(
            "Defekt nie narusza strefy krytycznej", "NOT_EVALUATED", None, defekt_in_critical_tolerance,
            "Nie podano critical_region - kryterium pominiete.",
        )
    else:
        frac = _frac(all_defekt, critical_set)
        if frac is None:
            c2 = CriterionResult(
                "Defekt nie narusza strefy krytycznej", "NOT_EVALUATED", None, defekt_in_critical_tolerance,
                "critical_region jest puste - kryterium pominiete.",
            )
        else:
            status = "PASS" if frac <= defekt_in_critical_tolerance else "FAIL"
            c2 = CriterionResult(
                "Defekt nie narusza strefy krytycznej", status, frac, defekt_in_critical_tolerance,
                f"{frac:.2f} strefy krytycznej dotkniete przez defekt (tolerancja {defekt_in_critical_tolerance}).",
            )

    # --- Kryterium 3: SKRĘT nie psuje orientacji domen krytycznych ---
    skret_set = set(int(i) for i in engine_result.get("skret_idx", []))
    if critical_region is None:
        c3 = CriterionResult(
            "Skret nie narusza orientacji domen krytycznych", "NOT_EVALUATED", None, skret_in_critical_tolerance,
            "Nie podano critical_region - kryterium pominiete.",
        )
    else:
        frac = _frac(skret_set, critical_set)
        if frac is None:
            c3 = CriterionResult(
                "Skret nie narusza orientacji domen krytycznych", "NOT_EVALUATED", None, skret_in_critical_tolerance,
                "critical_region jest puste - kryterium pominiete.",
            )
        else:
            status = "PASS" if frac <= skret_in_critical_tolerance else "FAIL"
            c3 = CriterionResult(
                "Skret nie narusza orientacji domen krytycznych", status, frac, skret_in_critical_tolerance,
                f"{frac:.2f} strefy krytycznej dotkniete przez skret (tolerancja {skret_in_critical_tolerance}).",
            )

    # --- Kryterium 4: REZONANS zgodny z funkcja (z Kroku 5) ---
    if mapping_result is None:
        c4 = CriterionResult(
            "Rezonans pokrywa sie z funkcja materialu", "NOT_EVALUATED", None, None,
            "Nie podano mapping_result (Krok 5) - kryterium pominiete.",
        )
    elif mapping_result.p_value is None:
        c4 = CriterionResult(
            "Rezonans pokrywa sie z funkcja materialu", "NOT_EVALUATED", None, None,
            "mapping_result nie da sie policzyc (pusty rezonans albo target_region).",
        )
    else:
        status = "PASS" if mapping_result.p_value < 0.05 else "FAIL"
        c4 = CriterionResult(
            "Rezonans pokrywa sie z funkcja materialu", status, mapping_result.p_value, 0.05,
            f"p={mapping_result.p_value:.4f} (test permutacyjny, Krok 5). {mapping_result.verdict_pl}",
        )

    criteria = [c1, c2, c3, c4]
    statuses = {c.status for c in criteria}
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "NOT_EVALUATED" in statuses:
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    lines = [f"[{c.status}] {c.name_pl}: {c.detail_pl}" for c in criteria]
    summary = (
        f"Status koncowy: {overall}.\n" + "\n".join(lines)
    )
    if overall == "INCOMPLETE":
        summary += (
            "\n\nUWAGA: INCOMPLETE oznacza, ze nie wszystkie kryteria mozna "
            "bylo sprawdzic (brak critical_region i/albo mapping_result) - "
            "to NIE jest to samo co PASS. Dostarcz brakujace dane, zeby "
            "uzyskac ostateczna odpowiedz."
        )

    return CloseoutReport(criteria=criteria, overall_status=overall, summary_pl=summary)
