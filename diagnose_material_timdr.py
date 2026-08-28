# diagnose_material_timdr.py
import traceback
from material_timdr import (
    RequirementsVector, design_material, PRIMARY_FUNCTIONS,
)

TEST_LATTICES = [
    ("2D", (8, 8)),
    ("3D", (4, 4, 4)),
]

def run_one(primary_function, lattice_size):
    req = RequirementsVector(
        primary_function=primary_function,
        temperature_range_c=(0, 100),
        environment="dry",
        notes=f"diag: {primary_function}, lattice={lattice_size}",
    )
    try:
        design_material(
            req,
            lattice_size=lattice_size,
            defect_atoms=None,
            dopant_atoms=None,
            target_region_atoms=None,
            critical_region_atoms=None,
            bond_length=1.0,
            n_permutations=100,
            seed=123,
        )
        print(f"[OK] {primary_function} {lattice_size}")
    except Exception as e:
        print(f"[FAIL] {primary_function} {lattice_size}: {e}")
        traceback.print_exc()

def main():
    for pf in PRIMARY_FUNCTIONS:
        for dim, size in TEST_LATTICES:
            run_one(pf, size)

if __name__ == "__main__":
    main()
