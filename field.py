"""
field.py — Krok 3: pole sygnału zakotwiczone w geometrii atomów.

Bierze Lattice (Krok 2) i produkuje SignalField: kilka nazwanych tablic
liczb, PO JEDNEJ WARTOŚCI NA ATOM, na których dalej pracuje spatial_timdr.py
(Krok 4). To jest dokładnie punkt, w którym geometria (fizyka/chemia)
zamienia się w "sygnał S" w sensie z timdr_core/core.py - tyle że nie
indeksowany czasem `t`, tylko indeksem atomu + grafem sąsiedztwa (Lattice.edges).

Cztery pola, wprost odpowiadające punktom z oryginalnego schematu ("lokalne
naprężenia, gęstość elektronowa, energia wiązań, defekty sieci"):

- bond_length_dev  — proxy naprężenia (odchylenie realnej długości wiązania
                      od nominalnej, uśrednione po sąsiadach atomu)
- bond_angle_dev   — proxy defektu sieci / energii odkształcenia kątowego
                      (odchylenie realnego kąta wiązania od kąta nominalnego
                      figury)
- orientation_deg  — lokalna orientacja domeny, TYLKO dla sieci 2D (honeycomb) -
                      substrat do wykrywania SKRĘTU (zmiana orientacji między
                      sąsiadującymi domenami). Liczona jako parametr
                      porządku orientacji wiązań (bond-orientational order
                      parameter Ψ_n, standardowa wielkość z fizyki materii
                      skondensowanej do wykrywania granic ziaren/domen w
                      sieciach 2D - patrz `_bond_orientational_angle_deg()`
                      niżej), NIE jako "kąt do pierwszego sąsiada" - ta
                      prostsza definicja została ODRZUCONA po znalezieniu
                      błędu: dla sieci dwupodsieciowej (honeycomb ma
                      podsieci A/B) taki kąt różni się o 60° MIĘDZY
                      podsieciami nawet w IDEALNEJ, bezdefektowej sieci
                      (podsieci A/B są przesunięte względem siebie o 60° -
                      to normalna cecha struktury krystalicznej, nie
                      defekt), co powodowało fałszywe wykrywanie "granicy
                      domeny" na KAŻDEJ krawędzi całej sieci
                      (test_spatial_timdr.py/test_validate.py złapały to
                      jako 100% pokrycie rezonansu niezależnie od
                      lokalizacji defektu). Ψ_n z n=2*koordynacja jest
                      niewrażliwy na ten podsieciowy artefakt (zweryfikowane
                      liczbowo - patrz test_field.py), a wciąż wykrywa
                      prawdziwe różnice orientacji między domenami.

                      DLA SIECI 3D (diamond) to pole NIE JEST liczone -
                      analogiczny niezmiennik dla sieci tetraedrycznej 3D
                      wymaga sferycznych parametrów porządku (Steinhardt
                      Q4/Q6), poza zakresem tego repo - patrz README,
                      sekcja "Zakres i ograniczenia". SpatialTIMDR pomija
                      wtedy SKRĘT (nie ma pola "orientation_deg" w params).
- dopant_proxy     — SYNTETYCZNY proxy gęstości elektronowej/domieszkowania:
                      suma gaussowskich "pagórków" wokół zadanych atomów
                      domieszki. Jawnie oznaczony jako proxy, NIE wynik
                      obliczeń DFT/kwantowych - patrz README.

Perturbacje (defect_atoms) przesuwają pozycje KOPII sieci o losowy wektor
(kontrolowany defect_strength * bond_length) - oryginalna Lattice z
lattice.py nigdy nie jest mutowana, żeby wielokrotne wywołania
build_signal_field() na tym samym obiekcie dawały niezależne, powtarzalne
(przy tym samym seed) wyniki.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from .lattice import Lattice
from .steinhardt import add_steinhardt_fields


@dataclass
class SignalField:
    """Krok 3 - nazwane pola sygnału + strefa docelowej funkcji materiału."""

    lattice: Lattice
    params: dict[str, np.ndarray]
    target_region: np.ndarray  # bool, len == lattice.n_atoms
    defect_atoms: list[int] = dc_field(default_factory=list)
    dopant_atoms: list[int] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.lattice.n_atoms
        if self.target_region.shape != (n,):
            raise ValueError(
                f"target_region ma kształt {self.target_region.shape}, oczekiwano ({n},)"
            )
        for name, arr in self.params.items():
            if arr.shape != (n,):
                raise ValueError(f"pole {name!r} ma kształt {arr.shape}, oczekiwano ({n},)")


def orientation_symmetry_deg(lattice: Lattice) -> float:
    """Okres zawinięcia orientacji: 360/(2*koordynacja) - PODWOJONA
    koordynacja, nie sama koordynacja. Zweryfikowane liczbowo
    (test_field.py) na honeycomb: przy zwykłym 360/koordynacja (=120° dla
    sp2) dwie podsieci A/B honeycombu (przesunięte względem siebie o 60° -
    to normalna cecha struktury, nie defekt) dają RÓŻNE wartości orientacji
    nawet w idealnej, bezdefektowej sieci, co zalewa skręt() fałszywymi
    wykryciami na każdej krawędzi. Z 360/(2*koordynacja) (=60° dla sp2) obie
    podsieci dają IDENTYCZNĄ wartość (zweryfikowane: Ψ_n z n=2*koordynacja
    jest niewrażliwy na przesunięcie o dokładnie pół okresu)."""
    n = lattice.figure.coordination
    return 360.0 / (2 * n) if n > 0 else 360.0


def _bond_orientational_angle_deg(positions: np.ndarray, atom_idx: int, neighbor_idx: list[int], n_fold: int) -> float:
    """Parametr porządku orientacji wiązań Ψ_n = (1/N)Σ exp(i·n·θ_k) -
    standardowa wielkość z fizyki materii skondensowanej do lokalnej
    orientacji sieci 2D, niewrażliwa na kolejność sąsiadów w liście (suma
    po WSZYSTKICH sąsiadach, nie wybór "pierwszego"). Zwraca
    arg(Ψ_n)/n_fold w stopniach, w zakresie który dalej zawijamy modulo
    orientation_symmetry_deg()."""
    if not neighbor_idx:
        return 0.0
    p0 = positions[atom_idx]
    thetas = [math.atan2(*(positions[j] - p0)[::-1]) for j in neighbor_idx]
    psi = sum(np.exp(1j * n_fold * t) for t in thetas) / len(thetas)
    if abs(psi) < 1e-12:
        return 0.0  # brak preferowanej orientacji (np. symetria sie skasowala)
    return math.degrees(np.angle(psi)) / n_fold


def build_signal_field(
    lattice: Lattice,
    defect_atoms: list[int] | None = None,
    defect_strength: float = 0.15,
    dopant_atoms: list[int] | None = None,
    dopant_amplitude: float = 1.0,
    dopant_sigma: float | None = None,
    target_region_atoms: list[int] | None = None,
    seed: int | None = None,
) -> SignalField:
    defect_atoms = list(defect_atoms or [])
    dopant_atoms = list(dopant_atoms or [])
    if target_region_atoms is None:
        target_region_atoms = dopant_atoms
    rng = np.random.default_rng(seed)

    # Pracujemy na NIEZALEŻNEJ kopii sieci - perturbacje pozycji nie mogą
    # wyciekać do oryginalnego obiektu Lattice przekazanego przez wywołującego.
    perturbed = Lattice(
        positions=lattice.positions.copy(),
        edges=list(lattice.edges),
        figure=lattice.figure,
        bond_length=lattice.bond_length,
    )
    dim = perturbed.positions.shape[1]
    for i in defect_atoms:
        disp = rng.normal(size=dim)
        norm = np.linalg.norm(disp)
        if norm > 0:
            disp = disp / norm
        disp = disp * defect_strength * lattice.bond_length
        perturbed.positions[i] = perturbed.positions[i] + disp

    n = perturbed.n_atoms
    bond_length_dev = np.zeros(n)
    bond_angle_dev = np.zeros(n)
    is_2d = dim == 2
    orientation_deg = np.zeros(n) if is_2d else None
    if is_2d:
        sym = orientation_symmetry_deg(lattice)
        n_fold = 2 * lattice.figure.coordination

    for i in range(n):
        lengths = perturbed.bond_lengths(i)
        if lengths:
            bond_length_dev[i] = float(np.mean([abs(l - lattice.bond_length) for l in lengths]))
        angles = perturbed.bond_angles_deg(i)
        if angles:
            bond_angle_dev[i] = float(np.mean([abs(a - lattice.figure.nominal_angle_deg) for a in angles]))
        if is_2d:
            nbrs = perturbed.neighbor_lists[i]
            raw_deg = _bond_orientational_angle_deg(perturbed.positions, i, nbrs, n_fold)
            orientation_deg[i] = raw_deg % sym

    dopant_proxy = np.zeros(n)
    if dopant_atoms:
        sigma = dopant_sigma if dopant_sigma is not None else 2.0 * lattice.bond_length
        for d in dopant_atoms:
            dvec = perturbed.positions - perturbed.positions[d]
            dist2 = np.sum(dvec * dvec, axis=1)
            dopant_proxy += dopant_amplitude * np.exp(-dist2 / (2.0 * sigma * sigma))

    target_region = np.zeros(n, dtype=bool)
    for i in target_region_atoms:
        target_region[i] = True

    params = {
        "bond_length_dev": bond_length_dev,
        "bond_angle_dev": bond_angle_dev,
        "dopant_proxy": dopant_proxy,
    }
    if is_2d:
        params["orientation_deg"] = orientation_deg
    else:
        # sieci 3D nie maja orientation_deg (patrz wyzej) - zamiast tego
        # dostaja Q4/Q6 (steinhardt.py), analogiczny co do roli substrat
        # dla anomalia()/defekt() w spatial_timdr.py, ale skalarny
        # (rotacyjnie niezmienniczy), nie katowy - NIE zastepuje skretu,
        # patrz README "Zakres i ograniczenia"
        add_steinhardt_fields(params, perturbed)

    return SignalField(
        lattice=perturbed,
        params=params,
        target_region=target_region,
        defect_atoms=defect_atoms,
        dopant_atoms=dopant_atoms,
    )
