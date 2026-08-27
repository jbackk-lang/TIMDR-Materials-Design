"""
mapping.py — Krok 5: mapowanie pól rezonansu na funkcję materiału.

NAJWAŻNIEJSZE zastrzeżenie całego repo jest tutaj, więc powtórzone wprost:
ten moduł sprawdza, czy strefy REZONANS==True (z spatial_timdr.py) pokrywają
się z zadaną `target_region` (gdzie CHCEMY funkcji) LEPIEJ NIŻ LOSOWO - przez
właściwy test permutacyjny z modelem null (dokładnie protokół z
timdr-signal-framework §13/§18: zdefiniuj metrykę PRZED zobaczeniem wyniku,
porównaj z modelem null, nie tylko surowe pokrycie).

To, co ten test MOŻE pokazać: czy pipeline (Kroki 2-4) jest WEWNĘTRZNIE
SPÓJNY - czyli czy zaprojektowane przez CIEBIE defekty/domieszki (Krok 3,
`defect_atoms`/`dopant_atoms`) faktycznie produkują sygnał REZONANS tam,
gdzie je umieściłeś. To NIE jest dowód, że "rezonans TIMDR" odpowiada
JAKIEJKOLWIEK realnej własności fizycznej materiału (przewodnictwu,
aktywności katalitycznej) - do tego potrzeba Kroku 7 (validate.py) na
ZMIERZONYCH danych prawdziwego materiału, czego ten moduł sam z siebie
nie dostarcza. Patrz README, sekcja "Zakres i ograniczenia".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MappingResult:
    n_atoms: int
    n_target: int
    n_rezonans: int
    observed_overlap: int
    precision: float | None  # observed_overlap / n_rezonans
    recall: float | None     # observed_overlap / n_target
    p_value: float | None    # permutacyjny, jednostronny: P(null_overlap >= observed)
    n_permutations: int
    verdict_pl: str

    SCOPE_DISCLAIMER_PL = (
        "Ten wynik mowi WYLACZNIE o wewnetrznej spojnosci pipeline'u "
        "(Kroki 2-4) na TEJ SYNTETYCZNEJ probce - NIE jest dowodem, ze "
        "rezonans TIMDR odpowiada jakiejkolwiek realnej wlasnosci fizycznej "
        "materialu. Do tego wymagany jest Krok 7 (validate.py) na "
        "ZMIERZONYCH danych prawdziwego materialu."
    )


def map_resonance_to_function(
    rezonans_idx: np.ndarray,
    target_region: np.ndarray,
    n_permutations: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> MappingResult:
    n = len(target_region)
    target_idx = np.where(target_region)[0]
    rez_set = set(int(i) for i in rezonans_idx)
    target_set = set(int(i) for i in target_idx)
    observed_overlap = len(rez_set & target_set)

    n_target = len(target_idx)
    n_rez = len(rez_set)

    if n_target == 0 or n_rez == 0:
        return MappingResult(
            n_atoms=n, n_target=n_target, n_rezonans=n_rez,
            observed_overlap=observed_overlap, precision=None, recall=None,
            p_value=None, n_permutations=0,
            verdict_pl=(
                "Nie da sie policzyc - target_region albo rezonans_idx jest "
                "puste (n_target={} n_rezonans={})".format(n_target, n_rez)
            ),
        )

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    null_overlaps = np.empty(n_permutations, dtype=int)
    for p in range(n_permutations):
        random_region = rng.choice(all_idx, size=n_target, replace=False)
        null_overlaps[p] = len(rez_set & set(int(i) for i in random_region))

    # standardowa korekta +1/+1 (nigdy p=0 z samej liczby permutacji)
    p_value = float((np.sum(null_overlaps >= observed_overlap) + 1) / (n_permutations + 1))
    precision = observed_overlap / n_rez
    recall = observed_overlap / n_target

    if p_value < alpha:
        verdict = (
            f"Pokrycie rezonans/target_region ({observed_overlap} atomow, "
            f"precision={precision:.2f} recall={recall:.2f}) jest istotnie "
            f"lepsze niz losowe rozmieszczenie tej samej wielkosci strefy "
            f"(p={p_value:.4f} < alpha={alpha}). {MappingResult.SCOPE_DISCLAIMER_PL}"
        )
    else:
        verdict = (
            f"Pokrycie rezonans/target_region ({observed_overlap} atomow, "
            f"precision={precision:.2f} recall={recall:.2f}) NIE jest "
            f"statystycznie odrozniane od losowego rozmieszczenia "
            f"(p={p_value:.4f} >= alpha={alpha}) - wroc do Kroku 2 lub 6 i "
            f"zmien geometrie/skład, zgodnie z Krokiem 5 procedury. "
            f"{MappingResult.SCOPE_DISCLAIMER_PL}"
        )

    return MappingResult(
        n_atoms=n, n_target=n_target, n_rezonans=n_rez,
        observed_overlap=observed_overlap, precision=precision, recall=recall,
        p_value=p_value, n_permutations=n_permutations, verdict_pl=verdict,
    )
