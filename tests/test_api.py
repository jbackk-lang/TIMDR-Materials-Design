"""
test_api.py — testy warstwy HTTP (material_timdr/api.py) przez FastAPI
TestClient (nie odpala prawdziwego serwera/portu). Sprawdza WYLACZNIE
poprawnosc "cienkiej warstwy" (routing, walidacja wejscia, serializacja
JSON) - logika TIMDR sama jest juz przetestowana w test_pipeline.py i
reszcie pakietu; te testy maja zlapac bledy typu "endpoint zwraca 500
zamiast 400" albo "serializacja gubi/psuje pole", nie duplikowac testow
pipeline'u.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from material_timdr.api import app

client = TestClient(app)


def test_root_serves_visual_ui_not_a_redirect_or_404():
    """GET / musi zwracac dzialajaca strone HTML (formularz + miejsce na
    SVG sieci), nie 404 (pierwsza wersja tego endpointu byla brakiem
    trasy w ogole) ani sam redirect na /docs (Swagger to surowe API, nie
    to, co uzytkownik chce zobaczyc po otwarciu przegladarki)."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Zaprojektuj materiał" in r.text
    assert "/design" in r.text  # strona faktycznie woła nasz endpoint API


def test_materials_endpoint_lists_all_five_real_presets():
    r = client.get("/materials")
    assert r.status_code == 200
    presets = r.json()["presets"]
    keys = {p["key"] for p in presets}
    assert keys == {"graphene", "h_bn", "diamond", "silicon", "germanium"}
    graphene = next(p for p in presets if p["key"] == "graphene")
    assert graphene["bond_length_angstrom"] == 1.42
    assert graphene["dimensionality"] == "2D"
    assert graphene["suggested_primary_function"] == "conductivity"


def test_root_ui_references_materials_endpoint():
    """Strona UI musi faktycznie wolac /materials (nie tylko /functions),
    inaczej lista przykladow bylaby martwym elementem formularza."""
    r = client.get("/")
    assert "/materials" in r.text


def test_design_with_graphene_preset_bond_length_reproduces_real_scale():
    """Uzycie prawdziwej dlugosci wiazania grafenu (1.42 A) przez /design
    powinno dac siec w tej samej skali - sprawdzone przez odleglosc
    miedzy dwoma sasiednimi atomami w zwroconych pozycjach (nie ufamy
    samej wartosci pola bond_length w odpowiedzi, mierzymy geometrie)."""
    payload = {
        "requirements": {"primary_function": "conductivity", "temperature_range_c": [-20, 80]},
        "lattice_size": [6, 6],
        "bond_length": 1.42,
        "seed": 1,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()
    a, b = body["lattice"]["edges"][0]
    pa, pb = body["lattice"]["positions"][a], body["lattice"]["positions"][b]
    dist = sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
    assert dist == pytest.approx(1.42, abs=1e-6)


def test_design_with_diamond_preset_bond_length_3d():
    payload = {
        "requirements": {"primary_function": "strength", "temperature_range_c": [0, 500]},
        "lattice_size": [3, 3, 3],
        "bond_length": 1.54,
        "seed": 1,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["lattice"]["bond_length"] == 1.54
    a, b = body["lattice"]["edges"][0]
    pa, pb = body["lattice"]["positions"][a], body["lattice"]["positions"][b]
    dist = sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
    assert dist == pytest.approx(1.54, abs=1e-6)


def test_design_can_reach_pass_via_api_with_widen_and_narrow_sigma_and_critical_region():
    """Regresja na dokladnie ten problem, ktory zglosil uzytkownik: bez
    critical_region_atoms i bez poszerzenia strefy docelowej UI/API bylo
    strukturalnie skazane na INCOMPLETE/FAIL. Ten test dowodzi, ze z
    kompletnymi, sensownymi danymi (widen=True, waskie dopant_sigma,
    podana strefa krytyczna) PASS jest realnie osiagalny przez /design,
    nie tylko przez wewnetrzne wywolanie design_material()."""
    payload = {
        "requirements": {"primary_function": "conductivity", "temperature_range_c": [0, 50]},
        "lattice_size": [10, 10],
        "dopant_atoms": [45],
        "dopant_amplitude": 1.0,
        "dopant_sigma": 0.6,
        "widen_target_to_dopant_neighbors": True,
        "critical_region_atoms": [0, 1, 2, 3],
        "seed": 3,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["closeout"]["overall_status"] == "PASS", body["closeout"]["summary_pl"]


def test_design_without_critical_region_or_widen_stays_incomplete_or_fail():
    """Dokumentuje ODWROTNA strone tego samego zachowania: bez tych
    danych wynik jest strukturalnie ograniczony do INCOMPLETE/FAIL - to
    JEST poprawne (kryteria 2/3 nie da sie ocenic bez critical_region),
    nie regresja. Test pilnuje, zeby to zachowanie pozostalo udokumentowane
    i swiadome, gdyby ktos pozniej "naprawil" to zmieniajac defaulty."""
    payload = {
        "requirements": {"primary_function": "conductivity", "temperature_range_c": [0, 50]},
        "lattice_size": [10, 10],
        "dopant_atoms": [45],
        "seed": 3,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["closeout"]["overall_status"] in ("INCOMPLETE", "FAIL")


def test_suggest_demo_params_returns_true_bulk_atom_not_edge():
    r = client.get("/suggest_demo_params", params={
        "primary_function": "strength", "lattice_size": "6,6,3", "bond_length": 1.54,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_atoms"] == 864
    assert body["n_bulk_atoms"] == 400  # patrz test_lattice.py::test_bulk_mask_diamond_excludes...
    assert len(body["critical_region"]) == 4
    assert body["suggested_dopant_sigma"] == pytest.approx(0.77)


def test_suggest_demo_params_then_design_reaches_pass_end_to_end():
    """Caly przeplyw, jakiego uzywa UI: /suggest_demo_params -> wypelnij
    pola -> POST /design. Musi realnie dawac PASS, nie tylko teoretycznie
    (to jest dokladnie to, co zglosil uzytkownik jako niedzialajace -
    wybor przykladu materialu i klikniecie 'Zaprojektuj' konczylo sie
    FAIL/INCOMPLETE z powodu zlego doboru atomu domieszki/strefy)."""
    suggestion = client.get("/suggest_demo_params", params={
        "primary_function": "strength", "lattice_size": "6,6,3", "bond_length": 1.54,
    }).json()

    payload = {
        "requirements": {"primary_function": "strength", "temperature_range_c": [0, 500]},
        "lattice_size": [6, 6, 3],
        "bond_length": 1.54,
        "dopant_atoms": [suggestion["dopant_atom"]],
        "dopant_amplitude": suggestion["suggested_dopant_amplitude"],
        "dopant_sigma": suggestion["suggested_dopant_sigma"],
        "widen_target_to_dopant_neighbors": True,
        "critical_region_atoms": suggestion["critical_region"],
        "n_permutations": 1000,
        "seed": 1,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["closeout"]["overall_status"] == "PASS", body["closeout"]["summary_pl"]


def test_suggest_demo_params_too_small_lattice_returns_400_not_500():
    r = client.get("/suggest_demo_params", params={
        "primary_function": "strength", "lattice_size": "1,1,1", "bond_length": 1.0,
    })
    assert r.status_code == 400


def test_design_too_large_3d_lattice_returns_400_fast_instead_of_hanging():
    """Regresja na zgloszony przez uzytkownika bug: zbyt duza siec 3D nie
    dawala bledu, tylko wisiala (steinhardt.py liczy Q4/Q6 nieprzerywalna,
    liniowa w n_atoms petla Pythona wolajaca scipy.special.sph_harm_y -
    zmierzone: 3200 atomow -> 25s, patrz komentarz przy MAX_ATOMS_3D w
    api.py). Ten test musi wrocic NATYCHMIAST z 400, nie liczyc nic."""
    payload = {
        "requirements": {"primary_function": "strength", "temperature_range_c": [0, 500]},
        "lattice_size": [50, 50, 50],  # 50*50*50*8 = 1 000 000 atomow, dalece nad limitem
    }
    import time
    t0 = time.time()
    r = client.post("/design", json=payload)
    elapsed = time.time() - t0
    assert r.status_code == 400
    assert "za duza" in r.json()["detail"]
    assert elapsed < 2.0, f"powinno odrzucic natychmiast, zajelo {elapsed:.2f}s"


def test_design_too_large_2d_lattice_returns_400_fast():
    payload = {
        "requirements": {"primary_function": "conductivity", "temperature_range_c": [-20, 80]},
        "lattice_size": [1000, 1000],  # 1000*1000*2 = 2 000 000 atomow
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 400
    assert "za duza" in r.json()["detail"]


def test_design_lattice_size_at_3d_limit_boundary_still_works():
    """Sanity check, ze limit nie jest ustawiony za nisko - siec dokladnie
    NA granicy (albo tuz pod nia) musi nadal dzialac normalnie, nie 400."""
    payload = {
        "requirements": {"primary_function": "strength", "temperature_range_c": [0, 500]},
        "lattice_size": [5, 5, 5],  # 5*5*5*8 = 1000 atomow, dobrze pod MAX_ATOMS_3D=2000
        "seed": 1,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200


def test_suggest_demo_params_too_large_lattice_returns_400_fast():
    import time
    t0 = time.time()
    r = client.get("/suggest_demo_params", params={
        "primary_function": "strength", "lattice_size": "50,50,50", "bond_length": 1.0,
    })
    elapsed = time.time() - t0
    assert r.status_code == 400
    assert "za duza" in r.json()["detail"]
    assert elapsed < 2.0


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_functions_endpoint_lists_all_primary_functions():
    r = client.get("/functions")
    assert r.status_code == 200
    functions = r.json()["primary_functions"]
    assert set(functions) == {"conductivity", "strength", "catalysis", "damping", "magnetism"}


def test_design_conductivity_2d_returns_full_result_shape():
    payload = {
        "requirements": {
            "primary_function": "conductivity",
            "temperature_range_c": [-20, 80],
        },
        "lattice_size": [6, 6],
        "dopant_atoms": [10],
        "dopant_amplitude": 2.0,
        "target_region_atoms": [10],
        "seed": 1,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()

    # najwazniejsze klucze najwyzszego poziomu obecne
    for key in ("requirements", "figure_suggestion", "synthesis_suggestion",
                "lattice", "field", "timdr", "mapping_result", "closeout"):
        assert key in body, f"brak klucza {key!r} w odpowiedzi /design"

    assert body["requirements"]["primary_function"] == "conductivity"
    assert body["figure_suggestion"]["base_figure"]["name"] == "sp2_planar"
    assert body["lattice"]["n_atoms"] == 72  # honeycomb 6x6, ta sama liczba co w innych testach
    assert len(body["lattice"]["positions"]) == body["lattice"]["n_atoms"]
    assert body["closeout"]["overall_status"] in ("PASS", "FAIL", "INCOMPLETE")
    assert isinstance(body["mapping_result"]["p_value"], (float, type(None)))
    # pole orientation_deg powinno byc obecne dla sieci 2D (sp2)
    assert "orientation_deg" in body["field"]["params"]
    assert "q4" not in body["field"]["params"]


def test_design_strength_3d_has_q4_q6_not_orientation():
    payload = {
        "requirements": {
            "primary_function": "strength",
            "temperature_range_c": [0, 500],
        },
        "lattice_size": [3, 3, 3],
        "seed": 1,
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["figure_suggestion"]["base_figure"]["name"] == "sp3_tetrahedral"
    assert "q4" in body["field"]["params"] and "q6" in body["field"]["params"]
    assert "orientation_deg" not in body["field"]["params"]


def test_design_invalid_primary_function_returns_400_not_500():
    payload = {
        "requirements": {
            "primary_function": "nie_istnieje",
            "temperature_range_c": [0, 10],
        },
        "lattice_size": [6, 6],
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 400
    assert "nie_istnieje" in r.json()["detail"]


def test_design_wrong_dimensionality_lattice_size_returns_400():
    payload = {
        "requirements": {
            "primary_function": "conductivity",  # sp2 -> potrzebuje 2D lattice_size
            "temperature_range_c": [0, 10],
        },
        "lattice_size": [3, 3, 3],
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 400


def test_design_out_of_range_atom_index_returns_400_not_500():
    payload = {
        "requirements": {
            "primary_function": "conductivity",
            "temperature_range_c": [0, 10],
        },
        "lattice_size": [4, 4],
        "dopant_atoms": [9999],  # daleko poza n_atoms dla malej sieci 4x4
    }
    r = client.post("/design", json=payload)
    assert r.status_code == 400


def test_design_is_reproducible_with_same_seed_via_api():
    payload = {
        "requirements": {
            "primary_function": "conductivity",
            "temperature_range_c": [0, 50],
        },
        "lattice_size": [6, 6],
        "dopant_atoms": [5],
        "seed": 42,
    }
    r1 = client.post("/design", json=payload)
    r2 = client.post("/design", json=payload)
    assert r1.json()["field"]["params"]["dopant_proxy"] == r2.json()["field"]["params"]["dopant_proxy"]
    assert r1.json()["mapping_result"]["p_value"] == r2.json()["mapping_result"]["p_value"]
