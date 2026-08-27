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

from fastapi.testclient import TestClient

from material_timdr.api import app

client = TestClient(app)


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
