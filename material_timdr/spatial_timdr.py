"""
spatial_timdr.py — Krok 4: TIMDR na polu materiału.

Generalizacja czterech operatorów anomalia/defekt/skręt/rezonans z
timdr_core/core.py (uniwersalny silnik tej rodziny repo, operujący na
szeregu czasowym S(t)) na SYGNAŁ NA GRAFIE PRZESTRZENNYM (atomy + wiązania
Lattice.edges), zamiast na sekwencji indeksowanej czasem. Ta sama
dyscyplina statystyczna co w oryginale - progi adaptacyjne z rozrzutu
DANYCH (p10-p90/MAD), z podłogą (floor_frac) na "prawie stałe" pola,
NIE progi uniwersalne wklejone na sztywno (patrz skill timdr-signal-framework,
§2).

Odpowiedniość operatorów (i gdzie DOKŁADNIE się różnią od oryginału):

| Oryginał (czasowy)                    | Tu (przestrzenny)                          |
|----------------------------------------|---------------------------------------------|
| anomalia: z-score względem populacji  | anomalia(): BEZ ZMIAN - populacja to wszystkie atomy, nie potrzebuje sąsiedztwa |
| defekt: skok między KOLEJNYMI próbkami | defekt(): skok między SĄSIADAMI W GRAFIE (Lattice.edges), nie kolejnością indeksów |
| skręt/twist: zmiana kierunku FLOW      | skret(): różnica orientacji między sąsiadami w grafie, z zawinięciem kątowym (mod symetria figury) |
| rezonans: >=min_count KANAŁÓW anomalii w tej samej CHWILI | rezonans(): >=min_count zbiorów wskazań (może być mix anomalia+defekt+skręt, nie tylko anomalia z różnych kanałów) w tym samym ATOMIE |

**ŚWIADOMA, UDOKUMENTOWANA różnica w rezonans()**: oryginalny
`TIMDRCore.rezonans()` w universal-state-analyzer liczy koincydencję
WYŁĄCZNIE między `anomalies()` z różnych kanałów (por. jego docstring).
Tu, zgodnie z krokiem 4 oryginalnego schematu użytkownika ("RESONANCE:
kombinacje powyższych"), `rezonans()` przyjmuje DOWOLNĄ mieszankę zbiorów
wskazań - anomalia+defekt+skręt razem, nie tylko anomalię. To jest
CELOWA, opisana tutaj różnica konwencji między tym modułem a resztą
ekosystemu, nie przypadkowy dryf (patrz skill timdr-signal-framework, §10,
"duplication-drift" - ten akapit istnieje właśnie po to, żeby taka różnica
nigdy nie była cicha).
"""
from __future__ import annotations

import numpy as np

from .field import SignalField
from .lattice import Lattice


def anomalia(values: np.ndarray, factor: float = 3.0, floor_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """MAD z-score względem CAŁEJ populacji atomów - ta sama logika co
    timdr_core.core.TIMDRCore.anomalies(), Z JEDNYM DODATKOWYM
    zabezpieczeniem: `mad == 0` NIE WYSTARCZY jako warunek fallbacku na
    idealnej (bezdefektowej) sieci, gdzie wszystkie wartości powinny być
    ZEROWE, ale w praktyce są rzędu 1e-16 (szum zaokrągleń
    zmiennoprzecinkowych z sumowania wektorów sieci w lattice.py) - MAD z
    takiego "prawie zerowego" pola bywa formalnie >0, ale NADAL tak małe,
    że pojedyncze ULP różnic między atomami dają astronomiczne z-score i
    fałszywie flagują idealną, bezdefektową sieć jako mającą anomalie
    (złapane w tests/test_spatial_timdr.py na honeycomb_lattice() bez
    żadnych defect_atoms - patrz _robust_spread() powyżej po ten sam
    wzorzec zastosowany do defekt()/skret())."""
    s = np.asarray(values, float)
    n = len(s)
    if n == 0:
        return np.array([], dtype=int), np.array([])
    scale = float(np.max(np.abs(s)))
    eps = max(scale, 1.0) * 1e-9
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826
    if mad <= eps or not np.isfinite(mad):
        std = np.std(s)
        mad = std if std > eps and np.isfinite(std) else max(abs(med) * floor_frac, 1e-9)
    z = (s - med) / mad
    idx = np.where(np.abs(z) > factor)[0]
    return idx, z


def _robust_spread(s: np.ndarray, floor_frac: float) -> float:
    """p90-p10, z DWUSTOPNIOWYM fallbackiem gdy to zero/niesk. - patrz
    skill timdr-signal-framework §2 ('pułapka zero-inflation'): pole
    materiałowe jest typowo w większości ZERO poza defektem/domieszką
    (mniej niż 10% atomów niezerowych), więc SAM p90-p10 (a nawet
    floor_frac*mediana, gdy mediana=0) kolabuje do ~0, co zamienia próg w
    praktyce w "dowolna niezerowa różnica = defekt" - stąd std(s) jako
    POŚREDNI fallback (odporny na zero-inflację, bo liczy się z całej
    populacji, nie z p10/p90 które przy >90% zer są równe 0), i dopiero
    potem floor_frac/stała jako ostatnia deska ratunku.

    UWAGA (znaleziony i naprawiony błąd): "> 0" NIE WYSTARCZY jako warunek
    odcięcia - przy >90% dokładnych zer i garstce małych niezerowych
    wartości, np.percentile() potrafi zwrócić FAŁSZYWIE DODATNI wynik
    rzędu 1e-16 (artefakt interpolacji między zerowym a niezerowym
    kwantylem), który jest ">0" formalnie, ale to szum numeryczny, nie
    sygnał - i to WŁAŚNIE ten artefakt (nie sam floor_frac) zalewał
    defekt()/skret() setkami fałszywych trafień w pierwszej wersji tego
    kodu (patrz test_validate.py). Epsilon jest więc liczony WZGLĘDEM
    SKALI DANYCH (max |wartość|), nie jako sztywna stała."""
    scale = float(np.max(np.abs(s))) if len(s) else 0.0
    eps = max(scale, 1.0) * 1e-9
    p10, p90 = np.percentile(s, 10), np.percentile(s, 90)
    spread = p90 - p10
    if spread > eps and np.isfinite(spread):
        return spread
    std = np.std(s)
    if std > eps and np.isfinite(std):
        return std
    return max(abs(np.median(s)) * floor_frac, 1e-9)


def defekt(values: np.ndarray, lattice: Lattice, factor: float = 0.3, floor_frac: float = 0.05) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Skok między SĄSIADAMI W GRAFIE (Lattice.edges) - próg z rozrzutu
    p90-p10 całego pola (patrz _robust_spread powyżej dla zero-inflacji).
    Zwraca (posortowane unikalne indeksy atomów incydentnych do 'zerwanej'
    krawędzi, {krawędź: |różnica|})."""
    s = np.asarray(values, float)
    n = len(s)
    if n < 2 or not lattice.edges:
        return np.array([], dtype=int), {}
    thr = factor * _robust_spread(s, floor_frac)

    edge_diffs: dict[tuple[int, int], float] = {}
    flagged: set[int] = set()
    for i, j in lattice.edges:
        d = abs(s[i] - s[j])
        edge_diffs[(i, j)] = float(d)
        if d > thr:
            flagged.add(i)
            flagged.add(j)
    return np.array(sorted(flagged), dtype=int), edge_diffs


def _circular_diff_deg(a: float, b: float, sym: float) -> float:
    """Odległość kątowa między dwoma orientacjami ZWINIĘTYMI modulo sym
    (patrz field._orientation_symmetry_deg) - min(|a-b|, sym-|a-b|)."""
    d = abs(a - b) % sym
    return min(d, sym - d)


def skret(orientation_deg: np.ndarray, lattice: Lattice, sym: float = 120.0, factor: float = 0.3, floor_frac: float = 0.05) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Różnica orientacji domeny między sąsiadami w grafie (odpowiednik
    'zmiany kierunku trendu' z oryginalnego TWIST, tu przestrzennie) - próg
    z rozrzutu p90-p10 KĄTOWYCH RÓŻNIC między wszystkimi krawędziami (nie
    samych wartości orientacji - inaczej figura symetryczna dałaby fałszywy
    rozrzut z samego zawinięcia mod sym), z podłogą."""
    s = np.asarray(orientation_deg, float)
    n = len(s)
    if n < 2 or not lattice.edges:
        return np.array([], dtype=int), {}
    all_diffs = np.array([_circular_diff_deg(s[i], s[j], sym) for i, j in lattice.edges])
    # _robust_spread liczy skale/epsilon z max(|all_diffs|) - dla rozkladu
    # roznic katowych to wlasciwa skala (te same zabezpieczenia zero-inflacji
    # i falszywie-dodatniego epsilon co w defekt(), patrz docstring _robust_spread)
    thr = factor * _robust_spread(all_diffs, floor_frac)

    edge_diffs: dict[tuple[int, int], float] = {}
    flagged: set[int] = set()
    for (i, j), d in zip(lattice.edges, all_diffs):
        edge_diffs[(i, j)] = float(d)
        if d > thr:
            flagged.add(i)
            flagged.add(j)
    return np.array(sorted(flagged), dtype=int), edge_diffs


def rezonans(index_lists: list[np.ndarray], n: int, min_count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Koincydencja >=min_count zbiorów wskazań W TYM SAMYM ATOMIE - patrz
    zastrzeżenie w docstringu modułu o różnicy względem timdr_core.rezonans()."""
    counts = np.zeros(n, dtype=int)
    for idxs in index_lists:
        counts[np.asarray(idxs, dtype=int)] += 1
    idx = np.where(counts >= min_count)[0]
    return idx, counts


class SpatialTIMDR:
    """Wygodny pipeline: SignalField -> anomalia/defekt/skręt/rezonans dla
    wszystkich pól naraz, analogicznie do TIMDRCore.analyze_multi()."""

    def __init__(
        self,
        anomaly_factor: float = 3.0,
        defekt_factor: float = 0.3,
        skret_factor: float = 0.3,
        rezonans_min: int = 2,
        floor_frac: float = 0.05,
    ) -> None:
        self.anomaly_factor = anomaly_factor
        self.defekt_factor = defekt_factor
        self.skret_factor = skret_factor
        self.rezonans_min = rezonans_min
        self.floor_frac = floor_frac

    def analyze(self, field: SignalField) -> dict:
        """UWAGA na generyczność: `defekt()` liczony jest dla KAŻDEGO pola w
        field.params (nie tylko bond_length_dev/bond_angle_dev z field.py) -
        to jest wymagane, żeby ten sam silnik dzialal zarówno na polach z
        build_signal_field() (Krok 3-4, nazwy z góry znane) jak i na
        DOWOLNIE nazwanych zmierzonych danych z validate.measured_field()
        (Krok 7, nazwy nieznane z góry - np. "conductivity_proxy").
        `skret()` jest WYJĄTKIEM w DRUGĄ STRONĘ - wymaga semantyki kątowej/
        kierunkowej (zawinięcie mod symetria figury), więc liczony jest
        TYLKO dla pola nazwanego dosłownie "orientation_deg" (jeśli
        obecne), a to pole jest SVIADOMIE WYŁĄCZONE z generycznej pętli
        `defekt()` poniżej: `defekt()` liczy zwykłą różnicę bezwzględną
        |a-b|, która dla pola kątowego zawiniętego mod sym (np. 119° vs 1°,
        realna odległość kątowa 2°) dałaby fałszywie ogromny skok (118) -
        DOKŁADNIE ten błąd złapał test_validate.py przy pierwszej wersji
        tego kodu (Jaccard=1.00 niezależnie od tego, gdzie faktycznie był
        defekt, bo cała siatka bazowa ma "pęknięcia" orientacji wynikające
        z samego zawinięcia, nie z żadnego wstrzykniętego defektu)."""
        n = field.lattice.n_atoms
        # UWAGA: 360/(2*koordynacja), zgodnie z field.orientation_symmetry_deg() -
        # NIE 360/koordynacja, patrz uzasadnienie tam (podsieci A/B).
        coord = field.lattice.figure.coordination
        sym = 360.0 / (2 * coord) if coord > 0 else 360.0

        anomaly_idx: dict[str, np.ndarray] = {}
        defekt_idx: dict[str, np.ndarray] = {}
        for name, vals in field.params.items():
            if name == "orientation_deg":
                # katowe/zawiniete - ANI anomalia() (MAD wzgledem populacji
                # nie rozumie zawiniecia mod sym) ANI defekt() (zwykla
                # roznica bezwzgledna) nie sa dla niego poprawne, patrz
                # skret() ponizej, ktory jest jedynym wlasciwym detektorem
                continue
            a_idx, _ = anomalia(vals, factor=self.anomaly_factor, floor_frac=self.floor_frac)
            anomaly_idx[name] = a_idx
            d_idx, _ = defekt(vals, field.lattice, factor=self.defekt_factor, floor_frac=self.floor_frac)
            defekt_idx[name] = d_idx

        if "orientation_deg" in field.params:
            skret_idx, _ = skret(
                field.params["orientation_deg"], field.lattice, sym=sym,
                factor=self.skret_factor, floor_frac=self.floor_frac,
            )
        else:
            skret_idx = np.array([], dtype=int)

        rez_sets = list(anomaly_idx.values()) + list(defekt_idx.values()) + [skret_idx]
        rez_idx, rez_counts = rezonans(rez_sets, n=n, min_count=self.rezonans_min)

        return dict(
            anomaly_idx=anomaly_idx,
            defekt_idx=defekt_idx,
            skret_idx=skret_idx,
            rezonans_idx=rez_idx,
            rezonans_counts=rez_counts,
        )
