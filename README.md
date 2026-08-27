# TIMDR-Materials-Design

Implementacja 8-krokowej procedury projektowania materiału od zera przy
użyciu TIMDR (anomalia/defekt/skręt/rezonans), zgodnej ze schematem
uzgodnionym z użytkownikiem. To jest **działający moduł kodu**, nie tylko
opis koncepcji — każdy krok ma odpowiadający mu plik w `material_timdr/`,
z testami weryfikującymi, że robi to, co deklaruje.

## Osiem kroków

| Krok | Co robi | Moduł |
|---|---|---|
| 1. Wektor wymagań | Funkcja materiału + warunki brzegowe, jako struktura danych | `requirements.py` |
| 2. Figury atomowe | sp2/sp3/liniowa, kąty policzone z geometrii, heurystyka funkcja→figura | `figures.py` |
| 3. Sieć + pole sygnału | Generatory sieci (honeycomb sp2, diamond sp3) + pola per-atom | `lattice.py`, `field.py` |
| 4. TIMDR na polu | anomalia/defekt/skręt/rezonans, generalizacja na graf przestrzenny | `spatial_timdr.py` |
| 5. Rezonans → funkcja | Test permutacyjny (null-model), nie surowe pokrycie | `mapping.py` |
| 6. Synteza | Heurystyki z literatury (temperatura/chłodzenie/ciśnienie) | `synthesis.py` |
| 7. Walidacja na pomiarach | Ten sam silnik na zmierzonych danych, porównanie z projektem | `validate.py` |
| 8. Zamknięcie | Checklist PASS/FAIL/NOT_EVALUATED na 4 kryteriach | `closeout.py` |

`pipeline.py::design_material()` spina kroki 1-6+8 w jedno wywołanie na
przykładzie syntetycznym. Krok 7 wymaga zewnętrznych danych pomiarowych,
więc jest wywoływany osobno — pełny przykład obu razem:
`examples/demo_graphene_dopant.py`.

## Szybki start

```bash
pip install -r requirements.txt
pytest -v                                  # 78 testów
PYTHONPATH=. python examples/demo_graphene_dopant.py
```

```python
from material_timdr import RequirementsVector, design_material

req = RequirementsVector(
    primary_function="conductivity",       # albo: strength, catalysis, damping, magnetism
    temperature_range_c=(-20, 85),
)
result = design_material(req, lattice_size=(8, 8), dopant_atoms=[42], seed=1)
print(result.closeout.summary_pl)
```

## Zakres i ograniczenia (przeczytaj przed użyciem)

To jest **metodologia zilustrowana na syntetycznych przykładach**, nie
zwalidowane narzędzie predykcyjne dla prawdziwych materiałów. Konkretnie:

**Co jest tu solidnie ugruntowane:**
- Geometria figur atomowych (kąt sp2=120°, sp3=arccos(-1/3)≈109.47°) jest
  policzona z definicji, nie wklejona z pamięci, i zweryfikowana na
  faktycznie wygenerowanych sieciach (`tests/test_lattice.py`).
- Generatory sieci (honeycomb, diamond) odtwarzają standardową,
  podręcznikową krystalografię — sprawdzone przez bezpośredni pomiar
  wygenerowanej geometrii (kąty, długości wiązań), nie przez zaufanie do
  wzoru.
- Test w Kroku 5 (mapping.py) to prawdziwy test permutacyjny z modelem
  null (ten sam protokół co w skillu `timdr-signal-framework` §13/§18) —
  nie surowe "ile się pokrywa", tylko "czy to więcej niż przypadek".
- Trzy realne błędy numeryczne znalezione i naprawione W TRAKCIE budowy
  tego repo (nie teoretyczne, złapane przez własne testy):
  1. Podsieci A/B sieci honeycomb są przesunięte o 60° nawet w idealnej,
     bezdefektowej sieci — naiwna definicja "orientacji" (kąt do
     pierwszego sąsiada) myliła to z granicą domeny na KAŻDEJ krawędzi.
     Naprawione parametrem porządku orientacji wiązań Ψ_n (n=2×koordynacja),
     standardową wielkością z fizyki materii skondensowanej.
  2. Zero-inflacja: pole materiałowe jest typowo >90% dokładnie zerem
     (poza defektem/domieszką), co zapadało próg `defekt()`/`skret()` do
     wartości rzędu maszynowego epsilon — każda różnica, nawet szum
     zaokrągleń, wychodziła jako "defekt". Naprawione wielostopniowym
     fallbackiem (p90-p10 → std → stała) z progiem odciecia względnym do
     skali danych.
  3. To samo zjawisko na jeszcze subtelniejszym poziomie: na idealnej
     sieci bez żadnych defektów, `anomalia()` potrafiła flagować atomy
     czysto z szumu zmiennoprzecinkowego (różnice rzędu 1e-16 dawały
     "trzysigmowe" wyniki, bo i licznik, i mianownik z-score były tego
     samego, astronomicznie małego rzędu wielkości).

**Czego to NIE dowodzi:**
- Że "rezonans TIMDR" odpowiada jakiejkolwiek realnej własności fizycznej
  materiału (przewodnictwu, aktywności katalitycznej, sile mechanicznej).
  Krok 5 sprawdza WEWNĘTRZNĄ SPÓJNOŚĆ pipeline'u (czy zaprojektowany przez
  ciebie defekt/domieszka produkuje sygnał tam, gdzie go umieściłeś) — nie
  jest to dowód na powiązanie z prawdziwą fizyką. Do tego służy Krok 7, na
  ZMIERZONYCH danych prawdziwego materiału, których to repo samo z siebie
  nie dostarcza.
- `FIGURE_TABLE` (Krok 2) i `SYNTHESIS_TABLE` (Krok 6) to tabele heurystyk
  z literatury materiałoznawczej, nie wynik obliczeń TIMDR — jawnie tak
  oznaczone w kodzie. Wpis dla magnetyzmu jest wyraźnie oznaczony jako
  najsłabiej ugruntowany: modeluje strukturę DOMENOWĄ (gdzie są granice
  domen, przez analogię do skrętu), nie mikroskopowe pochodzenie samego
  momentu magnetycznego (fizyka spinów d/f, poza zakresem tej geometrycznej
  ramy).
- Orientacja/skręt (Krok 4) jest zaimplementowana TYLKO dla sieci 2D
  (honeycomb/sp2). Analogiczny niezmiennik dla sieci 3D (diamond/sp3)
  wymagałby sferycznych parametrów porządku (Steinhardt Q4/Q6) — poza
  zakresem tego repo. `SpatialTIMDR` poprawnie pomija skręt dla sieci 3D
  (brak pola `orientation_deg`), nie udaje że go liczy.
- Demo (`examples/demo_graphene_dopant.py`) CELOWO kończy się statusem
  `FAIL` na Kroku 8, nie `PASS` — i to jest zamierzone, nie błąd: pokazuje
  rzeczywistą właściwość gładkiego "pagórka" domieszki (gaussowski
  rozkład), gdzie strefa `anomalia` jest szersza niż wąska strefa docelowa.
  Wynik nie został naciągnięty zmianą progów, żeby ładnie wyglądał w
  demo — dokładnie to (uczciwe raportowanie negatywnego/niejednoznacznego
  wyniku) jest standardem trzymanym w całym tym ekosystemie repozytoriów.

**Jedno zdanie podsumowania, uczciwie:** to repo automatyzuje PRZEPŁYW
informacji między ośmioma krokami projektowania materiału i dostarcza
prawdziwie działający, przetestowany silnik detekcji na dowolnej sieci
atomowej — ale decyzja, czy któryś krok odpowiada rzeczywistej fizyce
danego materiału, wymaga Kroku 7 na prawdziwych danych, których nikt jeszcze
tu nie podłączył.

## Struktura repo

```
material_timdr/
    requirements.py   — Krok 1: RequirementsVector
    figures.py         — Krok 2: figury atomowe (SP2_PLANAR, SP3_TETRAHEDRAL, SP_LINEAR)
    lattice.py          — Krok 2/3: generatory sieci (honeycomb_lattice, diamond_lattice)
    field.py            — Krok 3: pole sygnału (build_signal_field)
    spatial_timdr.py    — Krok 4: anomalia/defekt/skręt/rezonans na grafie (SpatialTIMDR)
    mapping.py          — Krok 5: test permutacyjny rezonans<->funkcja
    synthesis.py        — Krok 6: heurystyki syntezy
    validate.py          — Krok 7: walidacja na zmierzonych danych
    closeout.py          — Krok 8: checklist zamknięcia
    pipeline.py          — orkiestrator (design_material())
tests/                    — 78 testów pytest
examples/
    demo_graphene_dopant.py — pełny przebieg 8 kroków na jednym przykładzie
```
