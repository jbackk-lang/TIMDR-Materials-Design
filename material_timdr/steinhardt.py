"""
steinhardt.py — analog Kroku 3/4 dla sieci 3D (diamond/sp3): parametry
porządku Steinhardta Q4/Q6, odpowiednik Ψ_n z field.py dla sieci 2D
(honeycomb/sp2), ale zbudowany na harmonikach sferycznych zamiast
zwykłych kątów płaskich - to jest ISTOTNIE inna wielkość, nie ta sama
formuła w 3D, więc zasługuje na osobny moduł zamiast wciśnięcia w field.py.

Standardowa, ustalona wielkość z fizyki materii skondensowanej (Steinhardt,
Nelson, Ronchetti 1983) - używana do ilościowego opisu lokalnego porządku
(FCC vs HCP vs BCC vs amorficzny/ciekły) w symulacjach MD i analizie
koloidów/szkieł. TIMDR nie ma tu żadnej roli obliczeniowej - Q4/Q6 to
gotowa wielkość z literatury; TIMDR (anomalia/defekt, Krok 4) wchodzi
DOPIERO gdy Q4/Q6 staje się polem per-atom przepuszczanym przez
spatial_timdr, dokładnie tak samo jak bond_length_dev/bond_angle_dev.

Definicja:
    q_lm(i) = (1/N_b) * Σ_{j∈sąsiedzi(i)} Y_l^m(θ_ij, φ_ij)
    Q_l(i)  = sqrt( (4π/(2l+1)) * Σ_{m=-l}^{l} |q_lm(i)|² )

Q_l(i) jest SKALAREM, niezmienniczym względem obrotu całej sieci (bo sumuje
|q_lm|² po wszystkich m - standardowa własność, zweryfikowana bezpośrednio
w tests/test_steinhardt.py przez faktyczny obrót testowej sieci, nie
zakładana). To czyni go dobrym substratem dla anomalia()/defekt()
(populacyjny i sąsiedzki test odstawania), analogicznie jak
bond_length_dev/bond_angle_dev - ale Q4/Q6 NIE daje kąta domeny, więc NIE
zastępuje skret()/orientation_deg (patrz README, sekcja 'Zakres i
ograniczenia' - to uzupełnienie, nie pełny odpowiednik 2D).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import sph_harm_y

from .lattice import Lattice

DEGREES = (4, 6)


def _spherical_angles(vec: np.ndarray) -> tuple[float, float]:
    """(theta, phi) w konwencji scipy.special.sph_harm_y: theta=polarny
    (colatitude, [0,pi]), phi=azymutalny ([0,2pi])."""
    x, y, z = vec
    r = float(np.linalg.norm(vec))
    if r == 0:
        return 0.0, 0.0
    theta = math.acos(np.clip(z / r, -1.0, 1.0))
    phi = math.atan2(y, x) % (2 * math.pi)
    return theta, phi


def steinhardt_q(positions: np.ndarray, neighbor_lists: list[list[int]], atom_idx: int, l: int) -> float:
    """Q_l dla jednego atomu - patrz wzór w docstringu modułu."""
    nbrs = neighbor_lists[atom_idx]
    if not nbrs:
        return 0.0
    p0 = positions[atom_idx]
    total = 0.0
    for m in range(-l, l + 1):
        s = 0j
        for j in nbrs:
            theta, phi = _spherical_angles(positions[j] - p0)
            s += sph_harm_y(l, m, theta, phi)
        qlm = s / len(nbrs)
        total += abs(qlm) ** 2
    return math.sqrt((4 * math.pi / (2 * l + 1)) * total)


def steinhardt_field(lattice: Lattice, l: int) -> np.ndarray:
    """Q_l dla WSZYSTKICH atomów sieci naraz - zwraca tablicę (n_atoms,),
    gotową do wpięcia jako pole w SignalField.params (patrz field.py)."""
    n = lattice.n_atoms
    out = np.zeros(n)
    for i in range(n):
        out[i] = steinhardt_q(lattice.positions, lattice.neighbor_lists, i, l)
    return out


def add_steinhardt_fields(params: dict[str, np.ndarray], lattice: Lattice, degrees: tuple[int, ...] = DEGREES) -> None:
    """Dopisuje Ql pola (domyślnie q4, q6) do istniejącego słownika
    params w miejscu - wygodne wywołanie z field.build_signal_field() dla
    sieci 3D, gdzie orientation_deg nie jest liczone (patrz field.py)."""
    for l in degrees:
        params[f"q{l}"] = steinhardt_field(lattice, l)
