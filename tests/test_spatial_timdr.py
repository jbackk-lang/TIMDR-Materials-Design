import numpy as np
import pytest

from material_timdr.lattice import honeycomb_lattice, diamond_lattice
from material_timdr.field import build_signal_field
from material_timdr.spatial_timdr import anomalia, defekt, skret, rezonans, SpatialTIMDR


def _interior_atoms(lattice):
    return [i for i in range(lattice.n_atoms) if lattice.coordination(i) == lattice.figure.coordination]


def test_anomalia_flags_injected_outlier():
    values = np.zeros(50)
    values[10] = 20.0  # jawny outlier na tle stalego pola
    idx, z = anomalia(values, factor=3.0)
    assert 10 in idx
    assert len(idx) < 5, "anomalia nie powinna flagowac calej reszty tla"


def test_anomalia_population_mask_excludes_masked_atoms_from_threshold_and_flags():
    """population_mask=False atomy NIE moga wplywac na prog (mediana/MAD)
    ani zostac zaflagowane, nawet jesli same maja ekstremalna wartosc."""
    values = np.zeros(20)
    values[15] = 1000.0  # atom spoza populacji, ekstremalna wartosc
    values[3] = 5.0       # atom W populacji, umiarkowany outlier
    mask = np.ones(20, dtype=bool)
    mask[15] = False

    idx, z = anomalia(values, factor=3.0, population_mask=mask)
    assert 15 not in idx, "atom spoza populacji nie moze zostac zaflagowany"
    assert np.isnan(z[15]), "atom spoza populacji dostaje z=NaN, nie liczbe"
    # prog liczony TYLKO z populacji (bez 1000.0) - atom 3 powinien wypasc
    # jako outlier wzgledem reszty populacji (wszystkie zera poza nim)
    assert 3 in idx


def test_defekt_population_mask_skips_edges_touching_masked_atoms():
    hc = honeycomb_lattice(4, 4, bond_length=1.0)
    values = np.zeros(hc.n_atoms)
    boundary_atom = 0
    values[boundary_atom] = 1000.0  # atom "brzegowy" z ekstremalna wartoscia
    mask = np.ones(hc.n_atoms, dtype=bool)
    mask[boundary_atom] = False

    idx, edge_diffs = defekt(values, hc, population_mask=mask)
    assert boundary_atom not in idx
    # zadna krawedz dotykajaca boundary_atom nie powinna trafic do edge_diffs
    for (i, j) in edge_diffs:
        assert i != boundary_atom and j != boundary_atom


def test_diamond_ideal_lattice_q4_q6_have_zero_false_flags_after_boundary_fix():
    """Regresja na dokladnie ten bug: na idealnej (bez dopant/defect) sieci
    diamentowej, anomalia(q4)/defekt(q4) flagowaly atomy WYLACZNIE z powodu
    obciecia sasiedztwa na prawdziwej krawedzi skonczonej sieci (28 i 464
    atomow odpowiednio na diamond_lattice(6,6,3)) - po ograniczeniu
    populacji do Lattice.bulk_mask() w SpatialTIMDR.analyze() powinno to
    byc DOKLADNIE zero, nie 'mniej'."""
    dia = diamond_lattice(6, 6, 3, bond_length=1.54)
    field = build_signal_field(dia, seed=1)  # bez dopant_atoms/defect_atoms
    engine = SpatialTIMDR()
    result = engine.analyze(field)

    assert len(result["anomaly_idx"]["q4"]) == 0
    assert len(result["anomaly_idx"]["q6"]) == 0
    assert len(result["defekt_idx"]["q4"]) == 0
    assert len(result["defekt_idx"]["q6"]) == 0
    assert len(result["rezonans_idx"]) == 0


def test_anomalia_empty_input():
    idx, z = anomalia(np.array([]))
    assert len(idx) == 0


def test_anomalia_floor_guard_on_constant_field():
    values = np.zeros(30)
    idx, z = anomalia(values, factor=3.0, floor_frac=0.05)
    assert len(idx) == 0  # stale pole = brak anomalii, nie dzielenie przez 0


def test_defekt_flags_edge_with_injected_jump():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    interior = _interior_atoms(hc)
    values = np.zeros(hc.n_atoms)
    a = interior[len(interior) // 2]
    values[a] = 10.0  # skok tylko na jednym atomie -> jego krawedzie "peka"
    idx, diffs = defekt(values, hc, factor=0.3)
    assert a in idx
    for nb in hc.neighbor_lists[a]:
        assert nb in idx, "sasiad przez pekniete wiazanie tez powinien byc oflagowany"


def test_defekt_no_jump_no_flags():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    values = np.ones(hc.n_atoms) * 5.0  # stale pole, brak skokow
    idx, diffs = defekt(values, hc, factor=0.3)
    assert len(idx) == 0


def test_defekt_empty_lattice_edges():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    hc.edges = []
    idx, diffs = defekt(np.zeros(hc.n_atoms), hc)
    assert len(idx) == 0
    assert diffs == {}


def test_skret_flags_domain_boundary():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    orientation = np.zeros(hc.n_atoms)
    # polowa siatki (wiekszy indeks x) dostaje inna orientacje -> granica domen
    mid_x = np.median(hc.positions[:, 0])
    right_half = hc.positions[:, 0] > mid_x
    orientation[right_half] = 90.0
    idx, diffs = skret(orientation, hc, sym=120.0, factor=0.3)
    assert len(idx) > 0, "granica domen powinna zostac wykryta"
    # atomy DALEKO od granicy (oba skrajne rejony) NIE powinny byc flagowane
    leftmost = int(np.argmin(hc.positions[:, 0]))
    rightmost = int(np.argmax(hc.positions[:, 0]))
    assert leftmost not in idx
    assert rightmost not in idx


def test_skret_uniform_orientation_no_flags():
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    orientation = np.full(hc.n_atoms, 45.0)
    idx, diffs = skret(orientation, hc, sym=120.0, factor=0.3)
    assert len(idx) == 0


def test_circular_diff_handles_wraparound():
    from material_timdr.spatial_timdr import _circular_diff_deg
    # 119 i 1, zawiniete mod 120 - realna odleglosc katowa to 2, nie 118
    assert _circular_diff_deg(119.0, 1.0, sym=120.0) == pytest.approx(2.0)


def test_rezonans_requires_min_count_coincidence():
    n = 10
    idx, counts = rezonans([np.array([1, 2]), np.array([2, 3]), np.array([2, 5])], n=n, min_count=2)
    assert set(idx) == {2}
    assert counts[2] == 3


def test_rezonans_min_count_1_is_union():
    n = 10
    idx, counts = rezonans([np.array([1]), np.array([2])], n=n, min_count=1)
    assert set(idx) == {1, 2}


def test_spatial_timdr_analyze_end_to_end_on_field_with_defect_and_dopant():
    hc = honeycomb_lattice(8, 8, bond_length=1.0)
    interior = _interior_atoms(hc)
    defect_zone = interior[:3]
    f = build_signal_field(
        hc, defect_atoms=defect_zone, defect_strength=0.4,
        dopant_atoms=[interior[1]], dopant_amplitude=3.0, seed=5,
    )
    engine = SpatialTIMDR(rezonans_min=2)
    result = engine.analyze(f)
    assert "bond_length_dev" in result["anomaly_idx"]
    assert "dopant_proxy" in result["anomaly_idx"]
    assert len(result["rezonans_counts"]) == hc.n_atoms
    # przynajmniej jeden z atomow strefy defektu powinien pojawic sie w
    # ktoryms z zestawow wskazan (anomalia/defekt/skret) - to jest bardzo
    # slaby, ale wiarygodny sanity-check ze pipeline w ogole cos wykrywa
    all_flagged: set[int] = set()
    for name, idxs in result["defekt_idx"].items():
        all_flagged |= set(idxs)
    for name, idxs in result["anomaly_idx"].items():
        all_flagged |= set(idxs)
    assert all_flagged & set(defect_zone), "zaden detektor nie zlapal wstrzknietego defektu"


def test_spatial_timdr_analyze_works_on_arbitrary_named_measured_fields():
    """Silnik musi dzialac na polach o DOWOLNYCH nazwach (Krok 7 -
    validate.measured_field() moze przekazac cokolwiek), nie tylko na
    trzech nazwach znanych z field.py."""
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    n = hc.n_atoms
    values = np.zeros(n)
    interior = _interior_atoms(hc)
    values[interior[0]] = 50.0
    from material_timdr.field import SignalField
    f = SignalField(
        lattice=hc,
        params={"totally_custom_measured_name": values},
        target_region=np.zeros(n, dtype=bool),
    )
    engine = SpatialTIMDR(rezonans_min=1)
    result = engine.analyze(f)
    assert "totally_custom_measured_name" in result["defekt_idx"]
    assert interior[0] in result["defekt_idx"]["totally_custom_measured_name"]
    assert len(result["skret_idx"]) == 0  # brak orientation_deg w tym polu


def test_orientation_deg_excluded_from_generic_defekt_loop():
    """Regresja na blad opisany w spatial_timdr.SpatialTIMDR.analyze() -
    orientation_deg NIE moze trafic do defekt_idx (zawinienie kątowe psuje
    zwykla roznice bezwzgledna)."""
    hc = honeycomb_lattice(6, 6, bond_length=1.0)
    n = hc.n_atoms
    from material_timdr.field import SignalField
    orientation = np.zeros(n)
    orientation[::2] = 119.0  # sasiadujace atomy naprzemiennie ~0 i ~119
    f = SignalField(
        lattice=hc,
        params={"orientation_deg": orientation},
        target_region=np.zeros(n, dtype=bool),
    )
    engine = SpatialTIMDR()
    result = engine.analyze(f)
    assert "orientation_deg" not in result["defekt_idx"]
