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
| 4. TIMDR na polu | anomalia/defekt/skręt/rezonans, generalizacja na graf przestrzenny | `spatial_timdr.py`, `steinhardt.py` (Q4/Q6 dla sieci 3D) |
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
pytest -v                                  # 107 testów
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

## API (REST, lokalnie)

`material_timdr/api.py` wystawia `pipeline.design_material()` przez HTTP
(FastAPI) - cienka warstwa, nie dodaje żadnej nowej logiki TIMDR ponad to,
co jest opisane wyżej i przetestowane w `tests/test_pipeline.py`.

**Windows: dwuklik na `run.bat`** - tworzy `.venv`, instaluje zależności,
startuje serwer na `http://127.0.0.1:8000` (dokumentacja Swagger pod
`/docs`, generowana automatycznie z modeli Pydantic - nie może się
rozjechać z kodem). Zatrzymanie: Ctrl+C w oknie konsoli.

Ręcznie (Linux/macOS/Windows z Pythonem w PATH):
```bash
pip install -r requirements.txt
uvicorn material_timdr.api:app --host 127.0.0.1 --port 8000
```

Endpointy:
| Metoda | Ścieżka | Co robi |
|---|---|---|
| GET | `/health` | health check |
| GET | `/functions` | lista dozwolonych `primary_function` |
| POST | `/design` | pełny pipeline `design_material()`, zwraca JSON z figurą, siecią, polem, wynikami TIMDR, mapowaniem (Krok 5) i closeoutem (Krok 8) |

Przykład `POST /design`:
```bash
curl -X POST http://127.0.0.1:8000/design -H "Content-Type: application/json" -d '{
  "requirements": {"primary_function": "conductivity", "temperature_range_c": [-20, 80]},
  "lattice_size": [6, 6],
  "dopant_atoms": [10],
  "dopant_amplitude": 2.0,
  "target_region_atoms": [10],
  "seed": 1
}'
```

Błędne dane wejściowe (np. `primary_function` spoza `PRIMARY_FUNCTIONS`,
zły wymiar `lattice_size` względem figury sp2/sp3, indeks atomu poza
zakresem sieci) zwracają `400` z opisem błędu, nie `500` - sprawdzone w
`tests/test_api.py`. Krok 7 (walidacja na zmierzonych danych) NIE jest
wystawiony jako endpoint, bo z natury wymaga zewnętrznych danych
pomiarowych dostarczonych przez użytkownika - patrz `validate.py` i
`examples/demo_graphene_dopant.py`.

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
- Generatory sieci sprawdzone dodatkowo na PRAWDZIWYCH danych materiałowych
  (`tests/test_real_materials.py`): grafen (C-C 1.42 Å), azotek boru h-BN
  (B-N 1.45 Å), diament (C-C 1.54 Å), krzem (Si-Si 2.35 Å), german
  (Ge-Ge 2.45 Å) — długości wiązań to standardowe, ustalone stałe
  krystalograficzne (dane wejściowe, nie coś liczonego przez ten kod).
  Stała sieciowa wyliczona z tych długości zgadza się z powszechnie
  cytowaną wartością dla każdego materiału z dokładnością do ~1-2%
  (rozbieżność tego rzędu jest oczekiwana — model tu to sztywna,
  idealizowana geometria, nie symulacja DFT/MD z relaksacją sieci).
  Osobny test odtwarza znaną wakancję punktową w grafenie i w krzemie i
  potwierdza, że `SpatialTIMDR`/Q4/Q6 faktycznie ją wykrywają przy
  rzeczywistej skali długości (angstremy), nie tylko w bezwymiarowych
  jednostkach testowych.
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
  (honeycomb/sp2) — `SpatialTIMDR` poprawnie pomija skręt dla sieci 3D
  (brak pola `orientation_deg`), nie udaje że go liczy. Sieci 3D (diamond)
  dostają zamiast tego **Q4/Q6** (`steinhardt.py`) — standardowe parametry
  porządku Steinhardta z harmonik sferycznych, zweryfikowane bezpośrednio
  (nie z pamięci): stałe na całej idealnej sieci, rotacyjnie niezmiennicze
  (sprawdzone przez faktyczny obrót testowej sieci), i realnie różne w
  atomie sąsiadującym z defektem. **To NIE jest pełny odpowiednik
  orientacji/skrętu** — Q4/Q6 to skalar ("jak bardzo lokalne otoczenie
  wygląda jak idealne"), nie kąt domeny, więc trafiają do `anomalia()`/
  `defekt()` (jako kolejne pole per-atom, tak jak `bond_length_dev`), a
  NIE do `skret()`, który wciąż wymaga kierunkowej/kątowej semantyki
  dostępnej tylko w 2D.
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
    steinhardt.py        — Krok 4 (sieci 3D): Q4/Q6 z harmonik sferycznych
    mapping.py          — Krok 5: test permutacyjny rezonans<->funkcja
    synthesis.py        — Krok 6: heurystyki syntezy
    validate.py          — Krok 7: walidacja na zmierzonych danych
    closeout.py          — Krok 8: checklist zamknięcia
    pipeline.py          — orkiestrator (design_material())
    api.py               — REST API (FastAPI) nad pipeline.design_material()
tests/                    — 107 testów pytest (w tym test_real_materials.py: grafen, h-BN, diament, krzem, german; test_api.py: warstwa HTTP)
examples/
    demo_graphene_dopant.py — pełny przebieg 8 kroków na jednym przykładzie
run.bat                    — Windows: uruchamia API lokalnie (patrz sekcja "API" wyżej)
```
