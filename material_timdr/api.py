"""
api.py — lekkie REST API (FastAPI) nad pipeline'em design_material()
(Kroki 1-6+8, patrz pipeline.py). NIE dodaje żadnej nowej logiki TIMDR -
to czysto cienka warstwa HTTP wokół tego, co już jest przetestowane w
material_timdr/*.py i tests/test_pipeline.py.

Endpoints:
    GET  /                     — wizualny UI (przeglądarka): formularz +
                                  narysowana sieć atomowa (SVG) + wyniki
                                  TIMDR/mapping/closeout, patrz static/index.html
    GET  /health              — health check
    GET  /functions           — lista PRIMARY_FUNCTIONS (co można wpisać
                                  jako requirements.primary_function)
    POST /design               — uruchamia design_material() na podanych
                                  parametrach, zwraca pełny wynik jako JSON

Uruchomienie lokalne: patrz run.bat (Windows) albo
    uvicorn material_timdr.api:app --host 127.0.0.1 --port 8000
z katalogu głównego repo. Dokumentacja Swagger dostępna wtedy pod
http://127.0.0.1:8000/docs (FastAPI generuje ją automatycznie z modeli
Pydantic poniżej - nie jest pisana ręcznie, więc nie może się rozjechać
z faktycznym kodem).

UCZCIWOŚĆ ZAKRESU (to samo zastrzeżenie co w README, powtórzone tu, bo
API jest nowym punktem wejścia, który ktoś może przeczytać bez README):
to API uruchamia SYNTETYCZNY pipeline (Krok 7 - walidacja na zmierzonych
danych - nie jest tu wystawiony, bo z natury wymaga zewnętrznych danych
pomiarowych, patrz validate.py i examples/demo_graphene_dopant.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .requirements import RequirementsVector, PRIMARY_FUNCTIONS
from .figures import suggest_figure
from .lattice import honeycomb_lattice, diamond_lattice
from .pipeline import design_material, MaterialDesignResult
from .presets import REAL_MATERIAL_PRESETS

# Górny limit rozmiaru sieci - zabezpieczenie przed "zawieszeniem" serwera.
# POWÓD (zmierzone bezpośrednio, nie zgadywane): steinhardt.py liczy Q4/Q6
# (potrzebne tylko dla sieci 3D - strength/damping/magnetism) czystą pętlą
# Pythona wołającą scipy.special.sph_harm_y osobno dla każdego atomu x
# sąsiada x wartości m - to NIE jest zwektoryzowane, więc czas rośnie liniowo
# z liczbą atomów z dość dużym stałym narzutem: 2000 atomów (10x5x5) -> 12.6s,
# 3200 atomów (10x10x4) -> 25.2s (pełny design_material(), n_permutations=2000,
# zmierzone na tej maszynie). Serwer (uvicorn, jeden proces) blokuje się na
# czas liczenia - zbyt duża siec wpisana w UI wisiała bez końca i bez żadnego
# komunikatu (zgłoszone przez użytkownika). Limit trzyma worst-case w okolicach
# kilkunastu sekund zamiast dopuszczać wielominutowe/nieskończone zawieszenie.
# Sieci 2D (honeycomb) NIE mają Q4/Q6 (steinhardt.py w ogóle nie jest
# wołany) - zmierzone 20000 atomów (100x100) -> 7.5s, więc limit dla 2D
# jest wyżej.
MAX_ATOMS_3D = 2000
MAX_ATOMS_2D = 20000


def _check_lattice_size_limit(dimensionality: str, size: tuple[int, ...]) -> None:
    """Odrzuca za dużą siec PRZED jej wygenerowaniem/policzeniem pól (szybki
    400 zamiast wielosekundowego/wieloninutowego liczenia bez możliwości
    przerwania) - patrz uzasadnienie przy MAX_ATOMS_3D/MAX_ATOMS_2D wyżej."""
    if dimensionality == "2D":
        n1, n2 = size
        n_atoms = n1 * n2 * 2  # dwuatomowa baza honeycomb (A, B)
        limit = MAX_ATOMS_2D
    else:
        n1, n2, n3 = size
        n_atoms = n1 * n2 * n3 * 8  # osmioatomowa baza diamentowa
        limit = MAX_ATOMS_3D
    if n_atoms > limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Siec za duza: {n_atoms} atomow (limit {limit} dla sieci "
                f"{dimensionality}). Dla sieci 3D (strength/damping/magnetism) "
                "liczenie Q4/Q6 (steinhardt.py) rosnie w przyblizeniu liniowo "
                "z liczba atomow i NIE jest przerywalne w trakcie - zbyt duza "
                "siec zawiesza serwer na dlugo zamiast dac blad. Zmniejsz "
                "rozmiar sieci."
            ),
        )


app = FastAPI(
    title="TIMDR-Materials-Design API",
    description=(
        "REST API nad 8-krokowa procedura projektowania materialu przy "
        "uzyciu TIMDR (anomalia/defekt/skret/rezonans). Patrz README repo "
        "'TIMDR-Materials-Design' dla pelnego opisu zakresu i ograniczen "
        "metody - to API NIE dodaje nowej logiki, tylko wystawia "
        "pipeline.design_material() przez HTTP."
    ),
    version="1.0.0",
)

# CORS otwarty domyślnie (localhost-only use-case, uruchamiane przez
# run.bat na maszynie użytkownika) - jeśli API ma być wystawione poza
# localhost, zawęź allow_origins przed wdrożeniem.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Pydantic modele wejścia/wyjścia
# ---------------------------------------------------------------------
class RequirementsIn(BaseModel):
    primary_function: str = Field(
        ..., description=f"Jedna z: {', '.join(PRIMARY_FUNCTIONS)}"
    )
    temperature_range_c: tuple[float, float]
    pressure_range_pa: tuple[float, float] = (101325.0, 101325.0)
    environment: str = "dry"
    notes: str = ""


class DesignRequest(BaseModel):
    requirements: RequirementsIn
    lattice_size: list[int] = Field(
        default=[8, 8],
        description="(n1, n2) dla figur 2D/sp2 lub (n1, n2, n3) dla 3D/sp3",
    )
    defect_atoms: Optional[list[int]] = None
    defect_strength: float = 0.3
    dopant_atoms: Optional[list[int]] = None
    dopant_amplitude: float = 1.0
    dopant_sigma: Optional[float] = None
    target_region_atoms: Optional[list[int]] = None
    critical_region_atoms: Optional[list[int]] = None
    bond_length: float = 1.0
    n_permutations: int = 2000
    seed: Optional[int] = None
    widen_target_to_dopant_neighbors: bool = False


# ---------------------------------------------------------------------
# Serializacja MaterialDesignResult -> JSON-safe dict
# (ręcznie, nie dataclasses.asdict - wynik zawiera np.ndarray i zagnieżdżone
# dataclasses, które asdict nie zamieni poprawnie na typy JSON)
# ---------------------------------------------------------------------
def _idx_dict_to_lists(d: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: [int(i) for i in arr] for name, arr in d.items()}


def serialize_result(result: MaterialDesignResult) -> dict:
    lattice = result.field.lattice
    fig = result.figure_suggestion

    return {
        "requirements": result.requirements.as_dict(),
        "figure_suggestion": {
            "base_figure": {
                "name": fig.base_figure.name,
                "coordination": fig.base_figure.coordination,
                "nominal_angle_deg": fig.base_figure.nominal_angle_deg,
                "dimensionality": fig.base_figure.dimensionality,
            },
            "rationale_pl": fig.rationale_pl,
            "functional_zone_is_local_deviation": fig.functional_zone_is_local_deviation,
            "caveat_pl": fig.caveat_pl,
        },
        "synthesis_suggestion": {
            "favored_temperature_pl": result.synthesis_suggestion.favored_temperature_pl,
            "favored_cooling_pl": result.synthesis_suggestion.favored_cooling_pl,
            "favored_pressure_pl": result.synthesis_suggestion.favored_pressure_pl,
            "notes_pl": result.synthesis_suggestion.notes_pl,
        },
        "lattice": {
            "n_atoms": lattice.n_atoms,
            "n_edges": len(lattice.edges),
            "dimensionality": fig.base_figure.dimensionality,
            "bond_length": lattice.bond_length,
            "positions": lattice.positions.tolist(),
            "edges": [[int(a), int(b)] for a, b in lattice.edges],
        },
        "field": {
            "params": {name: arr.tolist() for name, arr in result.field.params.items()},
            "target_region": [int(i) for i in np.where(result.field.target_region)[0]],
            "defect_atoms": result.field.defect_atoms,
            "dopant_atoms": result.field.dopant_atoms,
        },
        "timdr": {
            "anomaly_idx": _idx_dict_to_lists(result.engine_result["anomaly_idx"]),
            "defekt_idx": _idx_dict_to_lists(result.engine_result["defekt_idx"]),
            "skret_idx": [int(i) for i in result.engine_result["skret_idx"]],
            "rezonans_idx": [int(i) for i in result.engine_result["rezonans_idx"]],
            "rezonans_counts": result.engine_result["rezonans_counts"].tolist(),
        },
        "mapping_result": {
            "n_atoms": result.mapping_result.n_atoms,
            "n_target": result.mapping_result.n_target,
            "n_rezonans": result.mapping_result.n_rezonans,
            "observed_overlap": result.mapping_result.observed_overlap,
            "precision": result.mapping_result.precision,
            "recall": result.mapping_result.recall,
            "p_value": result.mapping_result.p_value,
            "n_permutations": result.mapping_result.n_permutations,
            "verdict_pl": result.mapping_result.verdict_pl,
        },
        "closeout": {
            "overall_status": result.closeout.overall_status,
            "summary_pl": result.closeout.summary_pl,
            "criteria": [
                {
                    "name_pl": c.name_pl,
                    "status": c.status,
                    "value": c.value,
                    "tolerance": c.tolerance,
                    "detail_pl": c.detail_pl,
                }
                for c in result.closeout.criteria
            ],
        },
    }


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    # Prawdziwy wizualny UI (formularz + narysowana siec atomowa jako SVG,
    # nie tylko surowy JSON/Swagger) - to jest to, co ktos otwierajacy
    # http://127.0.0.1:8000 w przegladarce po uruchomieniu run.bat
    # faktycznie chce zobaczyc. Swagger nadal dostepny pod /docs (link w
    # naglowku strony), dla kogos kto chce wywolywac API programowo.
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/functions")
def functions() -> dict:
    return {"primary_functions": list(PRIMARY_FUNCTIONS)}


@app.get("/materials")
def materials() -> dict:
    """Gotowe ustawienia PRAWDZIWYCH materialow (grafen, h-BN, diament,
    krzem, german) - patrz presets.py dla zrodla danych i zastrzezen.
    UI (GET /) uzywa tego endpointu do wypelnienia listy przykladow, zeby
    liczby (dlugosci wiazan) byly zdefiniowane w jednym miejscu (presets.py),
    nie zdublowane w JS."""
    return {
        "presets": [
            {
                "key": p.key,
                "label_pl": p.label_pl,
                "bond_length_angstrom": p.bond_length_angstrom,
                "known_lattice_constant_angstrom": p.known_lattice_constant_angstrom,
                "dimensionality": p.dimensionality,
                "suggested_primary_function": p.suggested_primary_function,
                "note_pl": p.note_pl,
            }
            for p in REAL_MATERIAL_PRESETS.values()
        ]
    }


@app.get("/suggest_demo_params")
def suggest_demo_params(primary_function: str, lattice_size: str, bond_length: float = 1.0) -> dict:
    """Zwraca sensowny atom domieszki + strefe krytyczna DLA KONKRETNEJ
    sieci/funkcji, wyliczone na PRAWDZIWEJ wygenerowanej sieci przez
    Lattice.bulk_mask() (nie zgadywane po indeksie, nie sam warunek
    coordination(i)==figura.coordination).

    UZASADNIENIE (dwa oddzielne, po kolei znalezione bledy naiwnych
    heurystyk): (1) "srodkowy indeks = n_atoms // 2" jest ZLA - dla
    honeycomb_lattice(6,6) taki atom ma koordynacje 2 (brzeg), a dla
    diamond_lattice(6,6,3) koordynacje 1 (naroznik). (2) nawet filtr
    "coordination(i) == figura.coordination" NIE WYSTARCZA - na
    diamond_lattice(6,6,3) daje 605/864 atomow, z czego 205 wciaz dotyka
    PRAWDZIWEJ krawedzi (ma sasiada o niepelnej koordynacji), a pola
    liczone z geometrii sasiedztwa (Q4/Q6, steinhardt.py) sa wtedy mocno
    skazone efektem brzegowym niezaleznym od dopant_amplitude/sigma -
    patrz Lattice.bulk_mask() i spatial_timdr.py BOUNDARY_SENSITIVE_FIELDS
    po pelne uzasadnienie i zmierzone liczby. Ten endpoint uzywa wiec
    bulk_mask() (self I wszyscy sasiedzi w pelnej koordynacji), tej samej
    funkcji, ktorej teraz uzywa SpatialTIMDR wewnetrznie."""
    try:
        figure = suggest_figure(primary_function)
        size = tuple(int(x) for x in lattice_size.split(","))
        if figure.base_figure.dimensionality == "2D":
            if len(size) != 2:
                raise HTTPException(status_code=400, detail="Figura 2D wymaga lattice_size=n1,n2")
            _check_lattice_size_limit("2D", size)
            lattice = honeycomb_lattice(*size, bond_length=bond_length)
        else:
            if len(size) != 3:
                raise HTTPException(status_code=400, detail="Figura 3D wymaga lattice_size=n1,n2,n3")
            _check_lattice_size_limit("3D", size)
            lattice = diamond_lattice(*size, bond_length=bond_length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bulk = np.where(lattice.bulk_mask())[0].tolist()
    if not bulk:
        raise HTTPException(
            status_code=400,
            detail=(
                "Siec za mala - brak atomow 'bulk' (self i wszyscy sasiedzi w "
                "pelnej koordynacji), zwieksz rozmiar sieci."
            ),
        )
    dopant_atom = bulk[len(bulk) // 2]
    critical_region = bulk[:4] if len(bulk) >= 4 else bulk

    return {
        "n_atoms": lattice.n_atoms,
        "n_bulk_atoms": len(bulk),
        "dopant_atom": dopant_atom,
        "critical_region": critical_region,
        "suggested_dopant_amplitude": 1.0,
        "suggested_dopant_sigma": round(0.5 * bond_length, 6),
    }


@app.post("/design")
def design(req: DesignRequest) -> dict:
    try:
        requirements = RequirementsVector(
            primary_function=req.requirements.primary_function,
            temperature_range_c=req.requirements.temperature_range_c,
            pressure_range_pa=req.requirements.pressure_range_pa,
            environment=req.requirements.environment,
            notes=req.requirements.notes,
        )
        figure_for_limit = suggest_figure(requirements.primary_function)
        _check_lattice_size_limit(figure_for_limit.base_figure.dimensionality, tuple(req.lattice_size))
        result = design_material(
            requirements,
            lattice_size=tuple(req.lattice_size),
            defect_atoms=req.defect_atoms,
            defect_strength=req.defect_strength,
            dopant_atoms=req.dopant_atoms,
            dopant_amplitude=req.dopant_amplitude,
            dopant_sigma=req.dopant_sigma,
            target_region_atoms=req.target_region_atoms,
            critical_region_atoms=req.critical_region_atoms,
            bond_length=req.bond_length,
            n_permutations=req.n_permutations,
            seed=req.seed,
            widen_target_to_dopant_neighbors=req.widen_target_to_dopant_neighbors,
        )
    except ValueError as exc:
        # błędy walidacji RequirementsVector / złego rozmiaru sieci wzgledem
        # wymiarowosci figury (patrz pipeline.py) - to sa bledy UZYTKOWNIKA
        # (zle dane wejsciowe), nie awaria serwera, stad 400 nie 500
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexError as exc:
        # np. defect_atoms/dopant_atoms/target_region_atoms z indeksem poza
        # zakresem n_atoms danej sieci
        raise HTTPException(
            status_code=400, detail=f"Indeks atomu poza zakresem sieci: {exc}"
        ) from exc

    return serialize_result(result)
