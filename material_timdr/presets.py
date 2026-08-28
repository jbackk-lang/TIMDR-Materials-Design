"""
presets.py — gotowe ustawienia PRAWDZIWYCH materiałów do API/UI.

UCZCIWOŚĆ ŹRÓDŁA (to samo zastrzeżenie co w tests/test_real_materials.py,
skąd te same liczby pochodzą): `bond_length_angstrom` i
`known_lattice_constant_angstrom` to standardowe, szeroko cytowane stałe
krystalograficzne (dane WEJŚCIOWE), nie coś "odkrywanego" przez ten kod.
Ten moduł NIE dodaje żadnej nowej logiki TIMDR - to tylko wygodne stałe,
żeby ktoś klikający w UI (GET /) mógł zobaczyć wynik na realistycznej
skali długości (angstremy) zamiast bezwymiarowego bond_length=1.0, bez
przepisywania tych liczb ręcznie.

`suggested_primary_function` łączy materiał z jedną z PRIMARY_FUNCTIONS
WYŁĄCZNIE na podstawie geometrii sieci (sp2/2D -> conductivity/catalysis,
sp3/3D -> strength/damping/magnetism z figures.FIGURE_TABLE) - NIE jest to
twierdzenie, że dany materiał faktycznie jest w praktyce dobierany do tej
funkcji (patrz `note_pl` przy każdym wpisie, gdzie to nietrywialne - np.
h-BN jest izolatorem szerokoprzerwowym, nie przewodnikiem, mimo tej samej
geometrii sp2 co grafen; krzem/german są półprzewodnikami, nie materiałami
konstrukcyjnymi, mimo tej samej geometrii sp3 co diament).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialPreset:
    key: str
    label_pl: str
    bond_length_angstrom: float
    known_lattice_constant_angstrom: float
    dimensionality: str  # "2D" (honeycomb/sp2) | "3D" (diamond/sp3)
    suggested_primary_function: str
    note_pl: str


REAL_MATERIAL_PRESETS: dict[str, MaterialPreset] = {
    "graphene": MaterialPreset(
        key="graphene",
        label_pl="Grafen (C-C, sp2)",
        bond_length_angstrom=1.42,
        known_lattice_constant_angstrom=2.46,
        dimensionality="2D",
        suggested_primary_function="conductivity",
        note_pl=(
            "Podręcznikowy przykład sieci sp2/honeycomb - delokalizacja "
            "elektronów pi w płaszczyźnie, standardowa podstawa "
            "przewodnictwa w materiałach warstwowych."
        ),
    ),
    "h_bn": MaterialPreset(
        key="h_bn",
        label_pl="Azotek boru h-BN (B-N, sp2)",
        bond_length_angstrom=1.45,
        known_lattice_constant_angstrom=2.50,
        dimensionality="2D",
        suggested_primary_function="catalysis",
        note_pl=(
            "Ta sama geometria honeycomb co grafen, ale h-BN jest w "
            "rzeczywistości izolatorem szerokoprzerwowym (~6 eV), NIE "
            "przewodnikiem - stąd 'catalysis' (lokalne defekty/krawędzie), "
            "nie 'conductivity', jako mniej mylący domyślny wybór funkcji "
            "dla tej samej bazy sp2."
        ),
    ),
    "diamond": MaterialPreset(
        key="diamond",
        label_pl="Diament (C-C, sp3)",
        bond_length_angstrom=1.54,
        known_lattice_constant_angstrom=3.567,
        dimensionality="3D",
        suggested_primary_function="strength",
        note_pl=(
            "Podręcznikowy przykład sieci sp3/diamentopodobnej - "
            "izotropowe wiązania kowalencyjne, najtwardszy powszechnie "
            "znany materiał naturalny."
        ),
    ),
    "silicon": MaterialPreset(
        key="silicon",
        label_pl="Krzem (Si-Si, sp3)",
        bond_length_angstrom=2.35,
        known_lattice_constant_angstrom=5.431,
        dimensionality="3D",
        suggested_primary_function="strength",
        note_pl=(
            "Ta sama geometria sieci diamentowej co diament, ale krzem w "
            "praktyce jest półprzewodnikiem, nie materiałem "
            "konstrukcyjnym - 'strength' wybrane tu WYŁĄCZNIE ze względu "
            "na geometrię sp3, nie jako sugestia realnego zastosowania."
        ),
    ),
    "germanium": MaterialPreset(
        key="germanium",
        label_pl="German (Ge-Ge, sp3)",
        bond_length_angstrom=2.45,
        known_lattice_constant_angstrom=5.658,
        dimensionality="3D",
        suggested_primary_function="strength",
        note_pl=(
            "Analogicznie do krzemu: sieć diamentowa, ale german jest "
            "półprzewodnikiem - 'strength' to wybór geometryczny, nie "
            "funkcjonalny."
        ),
    ),
    "silicene": MaterialPreset(
        key="silicene",
        label_pl="Silicen (Si-Si, sp2, 2D)",
        bond_length_angstrom=2.23,
        known_lattice_constant_angstrom=3.87,
        dimensionality="2D",
        suggested_primary_function="conductivity",
        note_pl=(
            "2D odpowiednik krzemu (jak grafen dla węgla) - ALE prawdziwy "
            "silicen jest lekko pofalowany (buckling ~0.45 A), nie płaski "
            "jak grafen; ten model liczy tylko płaską sieć honeycomb, więc "
            "bond_length_angstrom dobrane tu tak, by odtworzyć znaną "
            "PŁASKĄ stałą sieciową a (nie literaturową odległość Si-Si "
            "3D ~2.28 A, która już zawiera efekt pofalowania i dałaby złą "
            "stałą sieciową w tym płaskim modelu). W praktyce silicen "
            "syntetyzowany był dotąd głównie na podłożach (np. Ag(111)), "
            "nie jako samodzielna, swobodna warstwa."
        ),
    ),
    "germanene": MaterialPreset(
        key="germanene",
        label_pl="Germanen (Ge-Ge, sp2, 2D)",
        bond_length_angstrom=2.29,
        known_lattice_constant_angstrom=3.97,
        dimensionality="2D",
        suggested_primary_function="conductivity",
        note_pl=(
            "2D odpowiednik germanu, analogicznie do silicenu wyżej - "
            "prawdziwy germanen też jest pofalowany (buckling ~0.65 A), "
            "ten płaski model tego nie odtwarza; bond_length_angstrom "
            "dobrane tak, by zgadzała się znana płaska stała sieciowa a, "
            "nie literaturowa (buckled) odległość Ge-Ge ~2.38-2.44 A. "
            "Również syntetyzowany głównie na podłożach, nie jako "
            "swobodna warstwa."
        ),
    ),
    "alpha_tin": MaterialPreset(
        key="alpha_tin",
        label_pl="Cyna szara α-Sn (Sn-Sn, sp3, 3D)",
        bond_length_angstrom=2.81,
        known_lattice_constant_angstrom=6.489,
        dimensionality="3D",
        suggested_primary_function="strength",
        note_pl=(
            "Jedyny czysty pierwiastek metaliczny (poza C/Si/Ge) o "
            "prawdziwej strukturze diamentowej - ale α-Sn (szara cyna) "
            "jest w praktyce KRUCHYM półmetalem/półprzewodnikiem "
            "zerowoprzerwowym, nie materiałem konstrukcyjnym; 'strength' "
            "to czysto geometryczny wybór jak przy krzemie/germanie. "
            "Ciekawostka historyczna: α-Sn jest stabilne tylko poniżej "
            "~13.2°C - powyżej tej temperatury cyna przechodzi w "
            "metaliczną odmianę β (biała cyna, struktura tetragonalna, "
            "NIE diamentowa) - efekt znany jako 'zaraza cynowa' "
            "('tin pest'), historycznie niszczący cynowe guziki/organy."
        ),
    ),
    "silicon_carbide": MaterialPreset(
        key="silicon_carbide",
        label_pl="Węglik krzemu 3C-SiC (Si-C, sp3, 3D)",
        bond_length_angstrom=1.89,
        known_lattice_constant_angstrom=4.3596,
        dimensionality="3D",
        suggested_primary_function="strength",
        note_pl=(
            "Struktura blendy cynkowej (zinc-blende) - topologicznie ta "
            "sama sieć co diament (koordynacja 4, te same kąty "
            "tetraedryczne), tylko z dwoma różnymi pierwiastkami "
            "naprzemiennie zamiast jednego - ten model generuje "
            "geometrię TYLKO jednorodną (jeden 'rodzaj' atomu), więc nie "
            "odróżnia Si od C w siatce, tylko odtwarza samą topologię "
            "wiązań Si-C. W odróżnieniu od krzemu/germanu/cyny wyżej, "
            "'strength' NIE jest tu czysto geometrycznym wyborem - SiC "
            "jest naprawdę używany konstrukcyjnie (ścierniwo, pancerze, "
            "elementy wysokotemperaturowe, twardość ~9-9.5 w skali Mohsa)."
        ),
    ),
}
