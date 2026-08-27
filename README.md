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
pytest -v                                  # 134 testów
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
startuje serwer na `http://127.0.0.1:8000`. Otwórz ten adres w
przeglądarce - `GET /` serwuje **wizualny UI** (`material_timdr/static/index.html`):
formularz (funkcja materiału, temperatura, rozmiar sieci, domieszka) +
narysowana sieć atomowa jako SVG (kolor = anomalia/defekt/skręt/rezonans,
przerywana obwódka = strefa docelowa) + wynik Kroku 5 (test permutacyjny)
i checklisty Kroku 8, aktualizowane po każdym kliknięciu "Zaprojektuj
materiał" - bez przeładowania strony. Dokumentacja Swagger dla wywołań
programowych dostępna osobno pod `/docs` (link w nagłówku UI), generowana
automatycznie z modeli Pydantic w `api.py`. Zatrzymanie serwera: Ctrl+C w
oknie konsoli.

**Limit rozmiaru sieci + przycisk "Anuluj":** `steinhardt.py` liczy Q4/Q6
(potrzebne tylko dla sieci 3D - strength/damping/magnesizm) czystą pętlą
Pythona (scipy `sph_harm_y` wołane osobno per atom×sąsiad×m) - NIE jest to
zwektoryzowane, więc czas rośnie w przybliżeniu liniowo z liczbą atomów i
NIE da się tego przerwać w trakcie liczenia. Zbyt duża siec wpisana w UI
potrafiła wisieć bardzo długo bez żadnego komunikatu i bez możliwości
powrotu do formularza (zgłoszone jako "zawieszony" interfejs). Naprawione
dwustronnie: (1) `/design` i `/suggest_demo_params` odrzucają za dużą sieć
NATYCHMIAST, zanim cokolwiek zacznie się liczyć (`MAX_ATOMS_3D=2000`,
`MAX_ATOMS_2D=20000` w `api.py`, dobrane na podstawie zmierzonych czasów:
2000 atomów 3D → ok. 12.6s, 3200 atomów 3D → ok. 25.2s, 20000 atomów 2D →
ok. 7.5s pełnego `design_material()`, ta sama maszyna); (2) UI ma przycisk
"Anuluj" obok "Zaprojektuj materiał" (AbortController) - przerywa
OCZEKIWANIE przeglądarki i od razu przywraca formularz do stanu
początkowego, nawet jeśli serwer sam w sobie nadal coś liczy w tle
(Python/uvicorn nie ma łatwego sposobu przerwania już trwającej,
synchronicznej pętli).

**Przykłady prawdziwych materiałów** - lista "Przykład materiału" w UI
(zasilana przez `GET /materials`, dane z `presets.py`, te same liczby co
w `tests/test_real_materials.py`) pozwala jednym kliknięciem ustawić
rzeczywistą długość wiązania i zobaczyć sieć w prawdziwej skali
(angstremy), zamiast bezwymiarowego `bond_length=1.0`:

| Materiał | Wiązanie | Długość (Å) | Geometria |
|---|---|---|---|
| Grafen | C-C | 1.42 | sp2 / honeycomb (2D) |
| Azotek boru h-BN | B-N | 1.45 | sp2 / honeycomb (2D) |
| Diament | C-C | 1.54 | sp3 / diamentowa (3D) |
| Krzem | Si-Si | 2.35 | sp3 / diamentowa (3D) |
| German | Ge-Ge | 2.45 | sp3 / diamentowa (3D) |

**Dlaczego wynik często wychodzi FAIL/INCOMPLETE, i jak dostać PASS:**
Krok 8 ma 4 kryteria - dwa z nich ("defekt"/"skręt nie naruszają strefy
krytycznej") są `NOT_EVALUATED` dopóki nie podasz pola "Strefa krytyczna"
w UI (bez tego status jest ograniczony do co najwyżej `INCOMPLETE`, nigdy
`PASS`). Kryterium "rezonans pokrywa się z funkcją" prawie zawsze wychodzi
`FAIL` (p=1.0), jeśli strefa docelowa to sam atom domieszki - gaussowski
"pagórek" domieszki ma ZEROWY gradient dokładnie w swoim szczycie (stąd
checkbox "Poszerz strefę docelową o sąsiadów domieszki", domyślnie
zaznaczony). Kryterium "anomalia tylko w strefie docelowej" zależy od
pola "Rozmycie domieszki (sigma)" - przy domyślnym sigmie (2x długość
wiązania) sygnał rozlewa się szerzej niż mała strefa docelowa niemal
zawsze; z węższym sigma (np. 0.5-0.7 dla bond_length=1.0) PASS jest
realnie osiągalny (sprawdzone w `tests/test_pipeline.py::test_narrow_dopant_sigma_with_widened_target_can_reach_pass`
i `tests/test_api.py::test_design_can_reach_pass_via_api_with_widen_and_narrow_sigma_and_critical_region`).
UI pokazuje żółty baner z wyjaśnieniem, KTÓRE dane brakują/dlaczego dane
kryterium nie przeszło, gdy status != PASS - to zazwyczaj uczciwie
zaraportowana właściwość modelu, nie oznaka błędu czy brakujących danych.
Zamiast ręcznie dobierać te wartości, UI woła `GET /suggest_demo_params`
przy starcie i po każdej zmianie funkcji/przykładu materiału - endpoint
liczy je na PRAWDZIWIE wygenerowanej sieci (nie na zgadywaniu po indeksie)
i wypełnia puste pola domieszki/strefy krytycznej/sigma automatycznie;
pola wypełnione ręcznie przez użytkownika nie są nadpisywane.

Wybór przykładu automatycznie ustawia `primary_function` na wartość
sensowną GEOMETRYCZNIE dla tej sieci (np. h-BN, krzem i german dostają
funkcję dobraną wyłącznie ze względu na tę samą geometrię co grafen/diament,
NIE dlatego że te materiały faktycznie są używane do tej funkcji w
praktyce - h-BN jest izolatorem, krzem/german są półprzewodnikami, nie
materiałami konstrukcyjnymi; pełne zastrzeżenie widoczne w UI po wyborze
i w `presets.py`).

Ręcznie (Linux/macOS/Windows z Pythonem w PATH):
```bash
pip install -r requirements.txt
uvicorn material_timdr.api:app --host 127.0.0.1 --port 8000
```

Endpointy:
| Metoda | Ścieżka | Co robi |
|---|---|---|
| GET | `/` | wizualny UI (formularz + SVG sieci + wyniki) |
| GET | `/health` | health check |
| GET | `/functions` | lista dozwolonych `primary_function` |
| GET | `/materials` | lista przykładów prawdziwych materiałów (grafen, h-BN, diament, krzem, german) z `presets.py` |
| GET | `/suggest_demo_params` | liczy na PRAWDZIWIE wygenerowanej sieci (nie zgaduje) sensowny atom domieszki + strefę krytyczną + sigma; UI wywołuje to automatycznie i wypełnia puste pola |
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
  4. Warunek brzegowy dla Q4/Q6 na sieciach 3D: filtrowanie atomów po
     samej koordynacji (`coordination(i) == 4`) NIE wystarcza, żeby
     wybrać prawdziwe atomy "bulk" (wewnętrzne) skończonej sieci
     diamentowej - atom może mieć pełną koordynację, a mimo to być
     bezpośrednio związany z atomem brzegowym (o niższej koordynacji),
     co silnie zniekształca sumę harmonik sferycznych Q4/Q6 (na idealnej,
     niezaburzonej sieci `diamond_lattice(6,6,3)` dawało to 28 fałszywych
     alarmów `anomalia()` i 464 fałszywych alarmów `defekt()` na polach
     q4/q6 - przy ZEROWEJ domieszce i ZEROWYM defekcie). Naprawione przez
     `Lattice.bulk_mask()` (atom I wszyscy jego sąsiedzi muszą mieć pełną
     koordynację) i parametr `population_mask` w `anomalia()`/`defekt()`,
     stosowany tylko do pól q4/q6 przez `SpatialTIMDR.BOUNDARY_SENSITIVE_FIELDS`
     - po poprawce: dokładnie 0 fałszywych alarmów (`tests/test_spatial_timdr.py::test_diamond_ideal_lattice_q4_q6_have_zero_false_flags_after_boundary_fix`).
     Sprawdzona alternatywa (wymaganie zgodności q4∧q6 przez `rezonans()`)
     NIE działa - zniekształcenie brzegowe jest niemal idealnie skorelowane
     między q4 i q6 (te same 28/464 atomów w obu polach), więc filtr
     koincydencji nic nie odsiewa.

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
  dostępnej tylko w 2D. Do niedawna sieci 3D były strukturalnie skazane na
  `FAIL` niezależnie od parametrów, z powodu błędu warunku brzegowego
  opisanego w punkcie 4 wyżej ("Co jest tu solidnie ugruntowane") - po
  poprawce `Lattice.bulk_mask()` PASS jest realnie osiągalny również dla
  diamentu (`tests/test_pipeline.py::test_diamond_dopant_scenario_can_reach_pass_after_boundary_condition_fix`).
- Demo (`examples/demo_graphene_dopant.py`) uruchamia DWA przebiegi na tej
  samej sieci/domieszce, jako jawna kontrola: **KONTROLA** (naiwna
  konfiguracja — target_region = sam atom domieszki, domyślne
  `dopant_sigma`) kończy się `FAIL` na Kroku 8 — to realna, sprawdzalna
  właściwość gładkiego gaussowskiego "pagórka" domieszki (jego dyskretny
  gradient jest zerowy dokładnie w szczycie, więc rezonans tworzy
  pierścień WOKÓŁ szczytu, nie sam szczyt), nie błąd kodu. **PO POPRAWCE**
  (ten sam atom domieszki, ale `widen_target_to_dopant_neighbors=True` +
  węższe `dopant_sigma`) kończy się `PASS` — z wydrukiem
  kryterium-po-kryterium pokazującym DOKŁADNIE, co się zmieniło między
  tymi dwoma przebiegami. Żaden z wyników nie został naciągnięty zmianą
  progów w `closeout.py` — obie konfiguracje używają tych samych,
  domyślnych tolerancji; różnica wynika wyłącznie z parametrów wejściowych
  (patrz `pipeline.py`, docstring `design_material()`, i sekcja niżej
  "Dlaczego wynik często wychodzi FAIL/INCOMPLETE, i jak dostać PASS").

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
    presets.py            — przykłady prawdziwych materiałów (grafen, h-BN, diament, krzem, german) do UI/API
    static/index.html     — wizualny UI serwowany pod GET / (formularz + SVG sieci + lista przykładów)
tests/                    — 134 testy pytest (w tym test_real_materials.py: grafen, h-BN, diament, krzem, german; test_api.py: warstwa HTTP + UI + presety + osiągalność PASS + limit rozmiaru sieci; test_lattice.py/test_spatial_timdr.py: Lattice.bulk_mask() i poprawka warunku brzegowego Q4/Q6)
examples/
    demo_graphene_dopant.py — pełny przebieg 8 kroków na jednym przykładzie
run.bat                    — Windows: uruchamia API lokalnie (patrz sekcja "API" wyżej)
```
