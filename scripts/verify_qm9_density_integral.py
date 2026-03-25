#!/usr/bin/env python3
"""
Verify whether the stored GPAW pseudo density integrates to the expected
electron count for QM9 molecules in the merged fp32 dataset.

This script focuses on the dataset layout used in:
    achmadjae/gpaw-qm9/dataset/gpaw_qm9_all_fp32.h5

By default it checks the first molecule group, which is expected to be:
    dsgdb9nsd_000001  (CH4)

What it computes
----------------
For one selected molecule group:
1. Reads `n_r` and metadata (`cell_angstrom`, `formula`, `num_atoms`)
2. Computes the simulation cell volume from `cell_angstrom`
3. Computes voxel volume in two ways:
   - full-cell convention:      dV = V_cell / (nx * ny * nz)
   - point-grid convention:     dV = V_cell / ((nx-1)*(ny-1)*(nz-1))
4. Integrates n_r over both conventions
5. Compares the result against:
   - total electrons from the chemical formula
   - valence electrons from the chemical formula

Notes
-----
- GPAW's `get_pseudo_density()` may correspond to pseudo/valence electrons,
  not necessarily total all-electron count.
- The full-cell convention is usually the first thing to test for a regular
  grid representation stored over the simulation cell.
- This script does not modify any file.

Usage
-----
Run with the gpaw conda environment, for example:

    conda run -n gpaw python gpaw-qm9/scripts/verify_qm9_density_integral.py

Optional arguments:

    conda run -n gpaw python gpaw-qm9/scripts/verify_qm9_density_integral.py \
        --h5-path gpaw-qm9/dataset/gpaw_qm9_all_fp32.h5 \
        --group dsgdb9nsd_000001

    conda run -n gpaw python gpaw-qm9/scripts/verify_qm9_density_integral.py \
        --index 0
"""

from __future__ import annotations

import argparse
import math
import re
from typing import Dict, List, Tuple

import h5py
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOTAL_ELECTRONS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}

VALENCE_ELECTRONS = {
    "H": 1,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether stored QM9 n_r integrates to expected electron count."
    )
    parser.add_argument(
        "--h5-path",
        default=str(PROJECT_ROOT / "dataset" / "gpaw_qm9_all_fp32.h5"),
        help="Path to merged fp32 QM9 HDF5 file.",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Explicit molecule group key to inspect, e.g. dsgdb9nsd_000001.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="0-based sorted group index to inspect if --group is not given.",
    )
    return parser.parse_args()


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def parse_formula(formula: str) -> List[Tuple[str, int]]:
    """
    Parse a simple chemical formula like CH4, H2O, C2H5N, etc.
    Returns list of (element, count).
    """
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula!r}")

    parsed = []
    for elem, count_str in tokens:
        count = int(count_str) if count_str else 1
        parsed.append((elem, count))
    return parsed


def electron_count_from_formula(formula: str, table: Dict[str, int]) -> int:
    total = 0
    for elem, count in parse_formula(formula):
        if elem not in table:
            raise KeyError(
                f"Element {elem!r} not supported by electron table. "
                f"Extend the table in this script."
            )
        total += table[elem] * count
    return total


def format_float(x: float) -> str:
    return f"{x:.10f}"


def relative_error(a: float, b: float) -> float:
    denom = max(abs(b), 1e-15)
    return abs(a - b) / denom


def main() -> None:
    args = parse_args()

    with h5py.File(args.h5_path, "r") as f:
        group_names = sorted(f.keys())
        if not group_names:
            raise RuntimeError("No molecule groups found in HDF5 file.")

        if args.group is not None:
            if args.group not in f:
                raise KeyError(f"Group {args.group!r} not found in {args.h5_path}")
            group_name = args.group
        else:
            if args.index < 0 or args.index >= len(group_names):
                raise IndexError(
                    f"--index {args.index} out of range for {len(group_names)} groups."
                )
            group_name = group_names[args.index]

        grp = f[group_name]
        n_r = np.asarray(grp["n_r"][...], dtype=np.float64)

        formula = decode_if_bytes(grp.attrs.get("formula", "UNKNOWN"))
        num_atoms = grp.attrs.get("num_atoms", None)
        cell = np.asarray(grp.attrs["cell_angstrom"], dtype=np.float64)

        nx, ny, nz = n_r.shape
        cell_volume = abs(float(np.linalg.det(cell)))

        dv_full_cell = cell_volume / (nx * ny * nz)
        dv_point_grid = cell_volume / max((nx - 1) * (ny - 1) * (nz - 1), 1)

        integral_full_cell = float(n_r.sum() * dv_full_cell)
        integral_point_grid = float(n_r.sum() * dv_point_grid)

        total_e = electron_count_from_formula(formula, TOTAL_ELECTRONS)
        valence_e = electron_count_from_formula(formula, VALENCE_ELECTRONS)

        print("=" * 80)
        print("QM9 DENSITY INTEGRAL VERIFICATION")
        print("=" * 80)
        print(f"HDF5 path                : {args.h5_path}")
        print(f"Group                    : {group_name}")
        print(f"Formula                  : {formula}")
        print(f"Num atoms                : {num_atoms}")
        print(f"n_r shape                : {n_r.shape}")
        print(f"n_r dtype (loaded)       : {n_r.dtype}")
        print()
        print("Cell information")
        print("-" * 80)
        print(f"cell_angstrom            :\n{cell}")
        print(f"cell volume (A^3)        : {format_float(cell_volume)}")
        print()
        print("Voxel conventions")
        print("-" * 80)
        print(f"dV full-cell             : {format_float(dv_full_cell)} A^3")
        print(f"dV point-grid            : {format_float(dv_point_grid)} A^3")
        print()
        print("Integrated density")
        print("-" * 80)
        print(f"Integral (full-cell)     : {format_float(integral_full_cell)}")
        print(f"Integral (point-grid)    : {format_float(integral_point_grid)}")
        print()
        print("Expected electron counts from formula")
        print("-" * 80)
        print(f"Total electrons          : {total_e}")
        print(f"Valence electrons        : {valence_e}")
        print()
        print("Comparison")
        print("-" * 80)
        print(
            f"|full-cell - total|      : {format_float(abs(integral_full_cell - total_e))}"
        )
        print(
            f"|full-cell - valence|    : {format_float(abs(integral_full_cell - valence_e))}"
        )
        print(
            f"|point-grid - total|     : {format_float(abs(integral_point_grid - total_e))}"
        )
        print(
            f"|point-grid - valence|   : {format_float(abs(integral_point_grid - valence_e))}"
        )
        print()
        print(
            f"relerr(full-cell, total)   : {format_float(relative_error(integral_full_cell, total_e))}"
        )
        print(
            f"relerr(full-cell, valence) : {format_float(relative_error(integral_full_cell, valence_e))}"
        )
        print(
            f"relerr(point-grid, total)  : {format_float(relative_error(integral_point_grid, total_e))}"
        )
        print(
            f"relerr(point-grid, valence): {format_float(relative_error(integral_point_grid, valence_e))}"
        )
        print("=" * 80)

        # Simple best-match summary
        candidates = {
            "full-cell vs total": abs(integral_full_cell - total_e),
            "full-cell vs valence": abs(integral_full_cell - valence_e),
            "point-grid vs total": abs(integral_point_grid - total_e),
            "point-grid vs valence": abs(integral_point_grid - valence_e),
        }
        best_name = min(candidates, key=candidates.get)
        best_err = candidates[best_name]

        print(f"Best numerical match     : {best_name}")
        print(f"Best absolute error      : {format_float(best_err)}")

        if math.isclose(best_err, 0.0, abs_tol=1e-6):
            print("Conclusion               : exact match within 1e-6")
        elif best_err < 1e-2:
            print("Conclusion               : very close match")
        else:
            print(
                "Conclusion               : not an exact electron-count match; inspect convention carefully"
            )


if __name__ == "__main__":
    main()
