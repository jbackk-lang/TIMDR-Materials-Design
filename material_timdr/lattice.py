"""
lattice.py — Krok 2/3: generatory syntetycznych sieci atomowych (baza
topologii) + budowa grafu wiązań (sąsiedztwo), na których dalej (field.py)
liczone jest pole sygnału.

Dwa generatory, dokładnie te wymienione explicite w oryginalnym schemacie
użytkownika:
- honeycomb_lattice()  — sp2/2D (trójkąty/plastry), jak w grafenie.
- diamond_lattice()    — sp3/3D (tetraedry), jak w sieci diamentu.

Obie geometrie są STANDARDOWE, ustalone w krystalografii - generatory tu
tylko odtwarzają znane wzory (wektory sieci + baza atomowa), nie proponują
niczego nowego. Poprawność (kąty/koordynacja) sprawdzona w tests/test_lattice.py
przez bezpośredni pomiar wygenerowanej geometrii, nie przez zaufanie do wzoru.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.spatial import cKDTree

from .figures import AtomicFigure, SP2_PLANAR, SP3_TETRAHEDRAL


@dataclass
class Lattice:
    """Sieć atomowa: pozycje + graf wiązań (krawędzie) + figura bazowa."""

    positions: np.ndarray          # (N, 2) albo (N, 3)
    edges: list[tuple[int, int]]   # nienaskierowane pary indeksów (i<j)
    figure: AtomicFigure
    bond_length: float
    neighbor_lists: list[list[int]] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.neighbor_lists:
            self.neighbor_lists = [[] for _ in range(len(self.positions))]
            for i, j in self.edges:
                self.neighbor_lists[i].append(j)
                self.neighbor_lists[j].append(i)

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    def coordination(self, i: int) -> int:
        return len(self.neighbor_lists[i])

    def bond_lengths(self, i: int) -> list[float]:
        """Aktualne (nie nominalne) długości wiązań atomu i do jego sąsiadów -
        liczone na BIEŻĄCYCH self.positions, więc odzwierciedlają perturbacje
        wprowadzone przez field.build_signal_field()."""
        p0 = self.positions[i]
        return [float(np.linalg.norm(self.positions[j] - p0)) for j in self.neighbor_lists[i]]

    def bond_angles_deg(self, i: int) -> list[float]:
        """Wszystkie kąty między parami wiązań wychodzącymi z atomu i,
        w stopniach. Puste/jednoelementowe sąsiedztwo -> []."""
        nbrs = self.neighbor_lists[i]
        if len(nbrs) < 2:
            return []
        p0 = self.positions[i]
        vecs = [self.positions[j] - p0 for j in nbrs]
        angles = []
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                va, vb = vecs[a], vecs[b]
                cos_t = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                cos_t = np.clip(cos_t, -1.0, 1.0)
                angles.append(math.degrees(math.acos(cos_t)))
        return angles


def _edges_from_distance(positions: np.ndarray, bond_length: float, tol: float = 0.15) -> list[tuple[int, int]]:
    """Buduje krawędzie łącząc pary atomów w odległości bond_length*(1±tol),
    przez KD-tree (szybkie nawet dla większych sieci niż potrzeba tutaj)."""
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=bond_length * (1 + tol))
    lo = bond_length * (1 - tol)
    edges = [(i, j) for (i, j) in pairs if np.linalg.norm(positions[i] - positions[j]) >= lo]
    return sorted(edges)


def honeycomb_lattice(n1: int = 6, n2: int = 6, bond_length: float = 1.0) -> Lattice:
    """Sieć plastra miodu (grafenopodobna, sp2/2D) - dwuatomowa baza (A,B)
    na siatce trójkątnej, n1 x n2 komórek elementarnych.

    Wektory sieci: a1=(1.5,√3/2)*b, a2=(1.5,-√3/2)*b (b=bond_length).
    Baza: A w (0,0) względem węzła komórki, B w (b,0)."""
    b = bond_length
    a1 = np.array([1.5 * b, math.sqrt(3) / 2 * b])
    a2 = np.array([1.5 * b, -math.sqrt(3) / 2 * b])
    basis_offset_B = np.array([b, 0.0])

    positions = []
    for i in range(n1):
        for j in range(n2):
            node = i * a1 + j * a2
            positions.append(node)               # atom A
            positions.append(node + basis_offset_B)  # atom B
    positions = np.array(positions)

    edges = _edges_from_distance(positions, bond_length)
    return Lattice(positions=positions, edges=edges, figure=SP2_PLANAR, bond_length=bond_length)


def diamond_lattice(n1: int = 3, n2: int = 3, n3: int = 3, bond_length: float = 1.0) -> Lattice:
    """Sieć diamentu (sp3/3D, tetraedryczna) - 8-atomowa baza w konwencjonalnej
    komórce sześciennej o boku ac = bond_length * 4/sqrt(3), powielona
    n1 x n2 x n3 razy.

    Standardowa baza diamentu (współrzędne ułamkowe komórki sześciennej):
    (0,0,0) (0,.5,.5) (.5,0,.5) (.5,.5,0) (.25,.25,.25) (.25,.75,.75)
    (.75,.25,.75) (.75,.75,.25)."""
    ac = bond_length * 4.0 / math.sqrt(3.0)
    basis_frac = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ])
    positions = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                cell_origin = np.array([i, j, k]) * ac
                for frac in basis_frac:
                    positions.append(cell_origin + frac * ac)
    positions = np.array(positions)

    edges = _edges_from_distance(positions, bond_length)
    return Lattice(positions=positions, edges=edges, figure=SP3_TETRAHEDRAL, bond_length=bond_length)
