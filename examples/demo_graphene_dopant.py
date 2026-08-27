"""
demo_graphene_dopant.py — pełny przebieg 8-krokowej procedury na jednym,
syntetycznym przykładzie: sieć sp2/2D (grafenopodobna) z domieszką w
konkretnym miejscu, celem = przewodnictwo w tym miejscu.

Uruchom: python examples/demo_graphene_dopant.py

To NIE jest symulacja prawdziwego grafenu ani prawdziwej fizyki
przewodnictwa - to demonstracja PRZEPŁYWU DANYCH przez wszystkie 8 kroków
procedury, na wygenerowanej syntetycznie sieci. Krok 7 (walidacja na
zmierzonych danych) jest tu zasymulowany przez traktowanie DRUGIEGO
niezależnego przebiegu build_signal_field() (inny seed) jako "pomiaru" -
w realnym użyciu measured_field() dostałby dane z prawdziwego eksperymentu.
"""
import numpy as np

from material_timdr import (
    RequirementsVector, design_material,
    validate_against_measurements, measured_field,
)
from material_timdr.lattice import honeycomb_lattice
from material_timdr.field import build_signal_field
from material_timdr.spatial_timdr import SpatialTIMDR


def main():
    print("=" * 70)
    print("TIMDR Material Design Protocol - demo: przewodnictwo (sp2)")
    print("=" * 70)

    # Krok 1: wektor wymagań
    req = RequirementsVector(
        primary_function="conductivity",
        temperature_range_c=(-20, 85),
        environment="dry",
        notes="Demo: domieszka lokalna, cel = przewodnictwo w miejscu domieszki",
    )
    print(f"\n[Krok 1] Wymagania: {req.as_dict()}")

    # Zbuduj sieć raz, żeby wybrać konkretny atom domieszki na podstawie
    # jej rzeczywistej geometrii (Krok 2/3 razem)
    hc = honeycomb_lattice(8, 8, bond_length=1.0)
    interior = [i for i in range(hc.n_atoms) if hc.coordination(i) == 3]
    dopant_atom = interior[len(interior) // 2]

    # target_region = domieszka + jej NAJBLIŻSI SĄSIEDZI, nie sam atom
    # domieszki. To jest CELOWA decyzja, nie kosmetyczne naciąganie wyniku:
    # dopant_proxy to gładki "pagórek" gaussowski (patrz field.py) - jego
    # DYSKRETNY GRADIENT (na czym opiera się defekt()) jest zerowy dokładnie
    # w SZCZYCIE pagórka (lokalne maksimum, symetryczne różnice się znoszą)
    # i największy na "zboczach" wokół niego. Pierwsza wersja tego demo
    # używała target_region=[sam_atom_domieszki] i dostawała FAIL na Kroku 5
    # (0% pokrycia) - poprawnie, bo rezonans faktycznie tworzy PIERŚCIEŃ
    # wokół szczytu, nie sam szczyt (zweryfikowane bezpośrednio: atom
    # szczytu nie trafia do rezonans_idx, ale WSZYSCY jego bezpośredni
    # sąsiedzi tak). To jest uczciwa, sprawdzalna właściwość geometrii
    # gradientu, nie błąd kodu - i jest dokładnie tym, co Krok 5 ma za
    # zadanie wykryć (rezonans w złym miejscu -> wróć i zmień geometrię).
    dopant_zone = [dopant_atom] + list(hc.neighbor_lists[dopant_atom])

    # Kroki 2-6 + 8, w jednym wywołaniu (Krok 7 osobno niżej)
    result = design_material(
        req,
        lattice_size=(8, 8),
        dopant_atoms=[dopant_atom],
        dopant_amplitude=2.5,
        target_region_atoms=dopant_zone,
        critical_region_atoms=interior[:5],  # przykladowa strefa "krytyczna"
        seed=7,
    )

    print(f"\n[Krok 2] Sugerowana figura: {result.figure_suggestion.base_figure.name} "
          f"({result.figure_suggestion.base_figure.nominal_angle_deg:.2f}°)")
    print(f"          Uzasadnienie: {result.figure_suggestion.rationale_pl}")

    print(f"\n[Krok 3] Sieć: {result.field.lattice.n_atoms} atomów, "
          f"{len(result.field.lattice.edges)} wiązań, pola: {list(result.field.params.keys())}")

    print(f"\n[Krok 4] Wyniki TIMDR:")
    for name, idx in result.engine_result["anomaly_idx"].items():
        print(f"          anomalia[{name}]: {len(idx)} atomów")
    for name, idx in result.engine_result["defekt_idx"].items():
        print(f"          defekt[{name}]: {len(idx)} atomów")
    print(f"          skret: {len(result.engine_result['skret_idx'])} atomów")
    print(f"          rezonans: {len(result.engine_result['rezonans_idx'])} atomów")

    print(f"\n[Krok 5] Mapowanie rezonans -> funkcja:")
    print(f"          {result.mapping_result.verdict_pl}")

    print(f"\n[Krok 6] Sugestia syntezy:")
    print(f"          Temperatura: {result.synthesis_suggestion.favored_temperature_pl}")
    print(f"          Chłodzenie:  {result.synthesis_suggestion.favored_cooling_pl}")
    print(f"          Uwagi:       {result.synthesis_suggestion.notes_pl}")

    # Krok 7: walidacja na "zmierzonych" danych - tu zasymulowana drugim,
    # niezależnym przebiegiem build_signal_field() (inny seed = inny szum
    # pomiarowy, ta sama lokalizacja domieszki - powinno dać wysoką zgodność)
    print(f"\n[Krok 7] Walidacja na 'zmierzonych' danych (symulacja):")
    measured_source = build_signal_field(
        hc, dopant_atoms=[dopant_atom], dopant_amplitude=2.5, seed=999,
    )
    measured = measured_field(hc, measured_source.params, target_region=result.field.target_region)
    engine = SpatialTIMDR()
    validation = validate_against_measurements(result.engine_result["rezonans_idx"], measured, engine=engine)
    print(f"          {validation.recommendation_pl}")

    print(f"\n[Krok 8] Zamknięcie projektu:")
    print(result.closeout.summary_pl)

    if result.closeout.overall_status != "PASS":
        print(
            "\n[Interpretacja] To demo CELOWO kończy się status!=PASS - i to jest "
            "pouczające, nie ukrywane. Kryterium 'anomalia tylko w target_region' "
            "nie przechodzi, bo strefa anomalia (MAD z-score na dopant_proxy) jest "
            "SZERSZA niż mała, 3-atomowa strefa docelowa - gaussowski 'pagórek' "
            "domieszki o tej amplitudzie/sigmie rozlewa się szerzej niż sam "
            "najbliższy sąsiedzki pierścień. Dokładnie do tego służy Krok 8: "
            "wykryć, że któraś strefa jest w złym miejscu/za szeroka, ZANIM "
            "uznasz projekt za zamknięty - zgodnie z Krokiem 5/6 procedury, "
            "następny krok to albo zawęzić dopant_amplitude/sigma (Krok 3), "
            "albo poszerzyć target_region do realistycznego rozmiaru strefy "
            "funkcjonalnej (Krok 1/5), nie wymuszać PASS zmianą progów."
        )


if __name__ == "__main__":
    main()
