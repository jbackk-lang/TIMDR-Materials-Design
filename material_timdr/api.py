"""
api.py — lekkie REST API (FastAPI) nad pipeline'em design_material()
(Kroki 1-6+8, patrz pipeline.py). NIE dodaje żadnej nowej logiki TIMDR -
to czysto cienka warstwa HTTP wokół tego, co już jest przetestowane w
material_timdr/*.py i tests/test_pipeline.py.

Endpoints:
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

from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .requirements import RequirementsVector, PRIMARY_FUNCTIONS
from .pipeline import design_material, MaterialDesignResult

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
    target_region_atoms: Optional[list[int]] = None
    critical_region_atoms: Optional[list[int]] = None
    bond_length: float = 1.0
    n_permutations: int = 2000
    seed: Optional[int] = None


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
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/functions")
def functions() -> dict:
    return {"primary_functions": list(PRIMARY_FUNCTIONS)}


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
        result = design_material(
            requirements,
            lattice_size=tuple(req.lattice_size),
            defect_atoms=req.defect_atoms,
            defect_strength=req.defect_strength,
            dopant_atoms=req.dopant_atoms,
            dopant_amplitude=req.dopant_amplitude,
            target_region_atoms=req.target_region_atoms,
            critical_region_atoms=req.critical_region_atoms,
            bond_length=req.bond_length,
            n_permutations=req.n_permutations,
            seed=req.seed,
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
