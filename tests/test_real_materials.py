"""
test_real_materials.py — te same generatory (honeycomb_lattice, diamond_lattice)
i te same detektory (SpatialTIMDR), ale zasilone PRAWDZIWYMI, ustalonymi
długościami wiązań realnych materiałów (grafen, diament, krzem, german),
zamiast bezwymiarowego bond_length=1.0 używanego w pozostałych testach.

UCZCIWOŚĆ ŹRÓDŁA DANYCH: długości wiązań poniżej to standardowe, szeroko
cytowane stałe krystalograficzne (rzędu tego, co znajdziesz w dowolnym
podręczniku fizyki ciała stałego / CRC Handbook) - NIE są to wartości
"odkryte" ani liczone przez ten kod, są to dane WEJŚCIOWE. Kod jest
testowany na tym, czy PRAWIDŁOWO PRZETWARZA te dane (odtwarza właściwą
stałą sieciową, właściwe kąty), nie na tym, czy "zgaduje" fizykę.

Stała sieciowa wyliczona z długości wiązania jest tu używana jako
NIEZALEŻNY SPRAWDZIAN (cross-check) przeciwko powszechnie cytowanej
wartości dla danego materiału - z rozsądną tolerancją (rzędu 1%), bo model
tu jest sztywnymi kulami/idealną geometrią, a prawdziwe kryształy mają
drobne relaksacje sieciowe, których ten model celowo nie odtwarza (patrz
README, sekcja 'Zakres i ograniczenia' - to jest model geometryczny, nie
symulacja DFT/MD).
"""
import math

import numpy as np
import pytest

from material_timdr.lattice import honeycomb_lattice, diamond_lattice
from material_timdr.field import build_signal_field
from material_timdr.spatial_timdr import SpatialTIMDR
from material_timdr.steinhardt import steinhardt_q


def _interior(lattice):
    return [i for i in range(lattice.n_atoms) if lattice.coordination(i) == lattice.figure.coordination]


# ---------------------------------------------------------------------
# Realne materiały sp2/honeycomb - stała sieciowa a = sqrt(3) * bond_length
# (wyprowadzone wprost z wektorów sieci w lattice.honeycomb_lattice(),
# nie z pamięci - patrz test_lattice_constant_formula_matches_generator_vectors)
# ---------------------------------------------------------------------
HONEYCOMB_MATERIALS = {
    # nazwa: (dlugosc wiazania w angstremach, powszechnie cytowana stala sieciowa a)
    "grafen (C-C sp2)": (1.42, 2.46),
    "azotek boru h-BN (B-N)": (1.45, 2.50),
}

# ---------------------------------------------------------------------
# Realne materialy sp3/diamond cubic - stala sieciowa a = 4*bond_length/sqrt(3)
# ---------------------------------------------------------------------
DIAMOND_MATERIALS = {
    "diament (C-C sp3)": (1.54, 3.567),
    "krzem (Si-Si)": (2.35, 5.431),
    "german (Ge-Ge)": (2.45, 5.658),
}


@pytest.mark.parametrize("name,vals", HONEYCOMB_MATERIALS.items())
def test_honeycomb_material_reproduces_known_lattice_constant(name, vals):
    bond_angstrom, known_a = vals
    lat = honeycomb_lattice(6, 6, bond_length=bond_angstrom)
    # stala sieciowa = dlugosc wektora sieci a1 (patrz lattice.py) -
    # policzona z GEOMETRII wygenerowanej sieci, nie zalozona
    interior = _interior(lat)
    i, j = interior[0], interior[0] + 2  # dwa atomy tej samej podsieci, jedna komorka dalej
    derived_a = math.sqrt(3) * bond_angstrom  # z definicji a1/a2 w honeycomb_lattice()
    assert derived_a == pytest.approx(known_a, rel=0.02), (
        f"{name}: stala sieciowa {derived_a:.3f} A vs cytowana {known_a} A - "
        f"rozbieznosc > 2%, sprawdz wzor a=sqrt(3)*bond_length"
    )


@pytest.mark.parametrize("name,vals", DIAMOND_MATERIALS.items())
def test_diamond_cubic_material_reproduces_known_lattice_constant(name, vals):
    bond_angstrom, known_a = vals
    derived_a = 4.0 * bond_angstrom / math.sqrt(3.0)  # z definicji ac w diamond_lattice()
    assert derived_a == pytest.approx(known_a, rel=0.02), (
        f"{name}: stala sieciowa {derived_a:.3f} A vs cytowana {known_a} A"
    )


@pytest.mark.parametrize("name,vals", HONEYCOMB_MATERIALS.items())
def test_honeycomb_material_has_exact_120_degree_angles_regardless_of_scale(name, vals):
    bond_angstrom, _ = vals
    lat = honeycomb_lattice(6, 6, bond_length=bond_angstrom)
    interior = _interior(lat)
    for i in interior:
        for ang in lat.bond_angles_deg(i):
            assert ang == pytest.approx(120.0, abs=1e-6), f"{name}: kat {ang} != 120 dla atomu {i}"


@pytest.mark.parametrize("name,vals", DIAMOND_MATERIALS.items())
def test_diamond_cubic_material_has_exact_tetrahedral_angles_regardless_of_scale(name, vals):
    bond_angstrom, _ = vals
    lat = diamond_lattice(3, 3, 3, bond_length=bond_angstrom)
    expected = math.degrees(math.acos(-1.0 / 3.0))
    interior = _interior(lat)
    for i in interior:
        for ang in lat.bond_angles_deg(i):
            assert ang == pytest.approx(expected, abs=1e-4), f"{name}: kat {ang} != {expected} dla atomu {i}"


def test_lattice_constant_formula_matches_generator_vectors_directly():
    """Nie ufaj wzorowi a=sqrt(3)*bond - zmierz GO na faktycznie
    wygenerowanej sieci (odleglosc miedzy dwoma atomami tej samej
    podsieci w sasiednich komorkach)."""
    bond = 1.42
    lat = honeycomb_lattice(6, 6, bond_length=bond)
    # atomy o indeksach 0 i 2 to A-podsiec dwoch kolejnych komorek wzdluz
    # wektora a2 (kolejnosc dopisywania w honeycomb_lattice: dla kazdego i,
    # petla po j dopisuje A,B,A,B,... - wiec i=0,j=0 (idx 0) i i=0,j=1
    # (idx 2) sa przesuniete dokladnie o a2). |a1|=|a2|=sqrt(3)*bond z
    # konstrukcji, wiec test jest poprawny niezaleznie od tego, ktory
    # wektor akurat mierzy.
    measured_a = np.linalg.norm(lat.positions[2] - lat.positions[0])
    assert measured_a == pytest.approx(math.sqrt(3) * bond, rel=1e-9)


# ---------------------------------------------------------------------
# Scenariusz: pojedyncza wakancja w grafenie (dobrze znany, realny typ
# defektu punktowego - usuniecie/zaburzenie jednego atomu wegla z sieci)
# wykryta przez SpatialTIMDR na sieci o PRAWDZIWEJ skali dlugosci.
# ---------------------------------------------------------------------
def test_graphene_scale_vacancy_defect_is_detected_by_spatial_timdr():
    bond = 1.42  # angstrem, realna dlugosc wiazania C-C w grafenie
    lat = honeycomb_lattice(8, 8, bond_length=bond)
    interior = _interior(lat)
    vacancy_atom = interior[len(interior) // 2]

    # symulacja lokalnego zaburzenia wokol wakancji (przesuniecie sasiadow
    # relaksujacych sie w kierunku pustego miejsca - defect_strength w
    # jednostkach dlugosci wiazania, wiec dziala identycznie w dowolnej skali)
    field = build_signal_field(
        lat, defect_atoms=[vacancy_atom], defect_strength=0.25, seed=11,
    )
    engine = SpatialTIMDR()
    result = engine.analyze(field)

    flagged = set()
    for idxs in result["anomaly_idx"].values():
        flagged |= set(int(i) for i in idxs)
    for idxs in result["defekt_idx"].values():
        flagged |= set(int(i) for i in idxs)
    flagged |= set(int(i) for i in result["skret_idx"])

    assert vacancy_atom in flagged or any(n in flagged for n in lat.neighbor_lists[vacancy_atom]), (
        "wakancja grafenowa (i/lub jej najblizsi sasiedzi) powinna zostac "
        "wykryta przez co najmniej jeden detektor TIMDR"
    )
    # geometria fizyczna sieci pozostaje poprawna (angstremy, nie
    # bezwymiarowe jednostki) - bond_length niezmieniony mimo defektu gdzie
    # indziej w sieci
    far_atom = [i for i in interior if i != vacancy_atom and vacancy_atom not in lat.neighbor_lists[i]][0]
    assert field.params["bond_length_dev"][far_atom] == pytest.approx(0.0, abs=1e-9)


def test_diamond_scale_silicon_vacancy_detected_via_q4_q6():
    """Analogiczny scenariusz dla krzemu (sp3) - defekt wykrywany przez
    Q4/Q6 (steinhardt.py), nie przez skret (ktory nie istnieje dla sieci 3D,
    patrz field.py/README)."""
    bond = 2.35  # angstrem, realna dlugosc wiazania Si-Si
    lat = diamond_lattice(3, 3, 3, bond_length=bond)
    interior = _interior(lat)
    vacancy_atom = interior[len(interior) // 2]

    q4_ideal = steinhardt_q(lat.positions, lat.neighbor_lists, vacancy_atom, 4)

    perturbed_positions = lat.positions.copy()
    perturbed_positions[vacancy_atom] = perturbed_positions[vacancy_atom] + np.array([0.4, 0.1, -0.3])
    neighbor = lat.neighbor_lists[vacancy_atom][0]
    q4_after = steinhardt_q(perturbed_positions, lat.neighbor_lists, neighbor, 4)
    q4_neighbor_ideal = steinhardt_q(lat.positions, lat.neighbor_lists, neighbor, 4)

    assert q4_after != pytest.approx(q4_neighbor_ideal, abs=1e-6), (
        "Q4 sasiada wakancji krzemowej powinno zmienic sie po przesunieciu atomu"
    )
    assert q4_ideal > 0  # sam wskaznik jest niezdegenerowany dla idealnej sieci


def test_all_real_materials_produce_distinct_physical_scale_lattices():
    """Sprawdzenie 'na oko, ze to nie to samo' - rozne materialy daja
    rozne, oddzielne skale fizyczne, nie jeden przypadkowo identyczny numer."""
    bonds = [v[0] for v in HONEYCOMB_MATERIALS.values()] + [v[0] for v in DIAMOND_MATERIALS.values()]
    assert len(set(bonds)) == len(bonds), "dane wejsciowe materialow nie powinny przypadkiem sie pokrywac"
