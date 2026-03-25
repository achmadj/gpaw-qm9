#!/usr/bin/env python3
"""
Create a focused GPAW verification run for CH4 from the QM9 source HDF5 and
compare pseudo-density vs all-electron density behavior.

Purpose
-------
This script is designed to answer two questions safely:

1. If we rerun GPAW for CH4 using the same settings as `run_gpaw_from_h5.py`,
   does the pseudo-density integral behave like the stored dataset result?
2. How do we obtain a density that integrates to the *total* number of electrons?

Key physics point
-----------------
- `calc.get_pseudo_density()` returns the PAW pseudo-density.
  It typically behaves like a valence-like density and does NOT have to integrate
  to the total all-electron count.
- `calc.get_all_electron_density()` reconstructs the all-electron density.
  That is the correct quantity to test against the total number of electrons.

Important clarification
-----------------------
`v_ext` is a potential, not a density. You should NOT add `v_ext` to `n(r)`.
To get total electron density, use:

    calc.get_all_electron_density(...)

not:

    n_r + v_ext

What this script does
---------------------
- Loads CH4 (`dsgdb9nsd_000001` by default) from the QM9 source HDF5
- Runs GPAW with the same essential settings as `run_gpaw_from_h5.py`
- Extracts:
    - pseudo-density with pad=True and pad=False
    - all-electron density with pad=True and pad=False
    - electrostatic potential
- Computes integrals using simple cell-volume / number-of-grid-points voxel volume
- Optionally compares the fresh pseudo-density with the stored merged fp32 dataset
- Saves all extracted arrays to an output HDF5 for inspection

Default files
-------------
Source QM9 geometry:
    gpaw-qm9/data/raw/qm9_smallest_20k.h5

Stored merged ML dataset:
    gpaw-qm9/dataset/gpaw_qm9_all_fp32.h5

Output:
    gpaw-qm9/gpaw_analysis_outputs/ch4_test/ch4_gpaw_density_modes.h5

Usage
-----
Run with the gpaw conda environment, for example:

    conda run -n gpaw python gpaw-qm9/scripts/test_ch4_gpaw_density_modes.py

Optional:

    conda run -n gpaw python gpaw-qm9/scripts/test_ch4_gpaw_density_modes.py \
        --mol-key dsgdb9nsd_000001 \
        --gridrefinement 2 \
        --out-path gpaw-qm9/gpaw_analysis_outputs/ch4_test/ch4_gpaw_density_modes.h5
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW

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


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPAW on CH4/QM9 and compare pseudo vs all-electron density integrals."
    )
    parser.add_argument(
        "--db-path",
        default=str(PROJECT_ROOT / "dataset" / "raw" / "qm9_smallest_20k.h5"),
        help="Path to the source QM9 HDF5 geometry database.",
    )
    parser.add_argument(
        "--stored-path",
        default=str(PROJECT_ROOT / "dataset" / "gpaw_qm9_all_fp32.h5"),
        help="Path to the stored merged fp32 dataset for comparison.",
    )
    parser.add_argument(
        "--mol-key",
        default="dsgdb9nsd_000001",
        help="QM9 molecule key to run. Default is CH4.",
    )
    parser.add_argument(
        "--out-path",
        default=str(PROJECT_ROOT / "gpaw_analysis_outputs" / "ch4_test" / "ch4_gpaw_density_modes.h5"),
        help="Output HDF5 path for fresh GPAW arrays and metadata.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_ROOT / "gpaw_analysis_outputs" / "ch4_test" / "logs" / "gpaw"),
        help="Directory for GPAW text logs.",
    )
    parser.add_argument(
        "--padding-bohr",
        type=float,
        default=4.0,
        help="Vacuum padding in Bohr, matching run_gpaw_from_h5.py.",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.2,
        help="Grid spacing in Angstrom, matching run_gpaw_from_h5.py.",
    )
    parser.add_argument(
        "--xc",
        default="LDA",
        help="Exchange-correlation functional.",
    )
    parser.add_argument(
        "--mode",
        default="fd",
        help="GPAW mode.",
    )
    parser.add_argument(
        "--gridrefinement",
        type=int,
        default=2,
        choices=[1, 2, 4],
        help="Grid refinement for all-electron density.",
    )
    parser.add_argument(
        "--skip-stored-compare",
        action="store_true",
        help="Skip comparison against stored fp32 merged dataset.",
    )
    return parser.parse_args()


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def parse_formula(formula: str) -> List[Tuple[str, int]]:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula!r}")
    out = []
    for elem, count_str in tokens:
        out.append((elem, int(count_str) if count_str else 1))
    return out


def electron_count_from_formula(formula: str, table: Dict[str, int]) -> int:
    total = 0
    for elem, count in parse_formula(formula):
        if elem not in table:
            raise KeyError(f"Element {elem!r} not found in electron table.")
        total += table[elem] * count
    return total


def cell_volume_ang3(cell: np.ndarray) -> float:
    return abs(float(np.linalg.det(np.asarray(cell, dtype=np.float64))))


def dv_full_cell(cell: np.ndarray, shape: Tuple[int, int, int]) -> float:
    nx, ny, nz = shape
    return cell_volume_ang3(cell) / (nx * ny * nz)


def integrate_density(density: np.ndarray, cell: np.ndarray) -> float:
    density64 = np.asarray(density, dtype=np.float64)
    dv = dv_full_cell(cell, density64.shape)
    return float(density64.sum() * dv)


def summarize_array(name: str, arr: np.ndarray, cell: np.ndarray) -> Dict[str, object]:
    arr64 = np.asarray(arr, dtype=np.float64)
    return {
        "name": name,
        "shape": tuple(int(x) for x in arr64.shape),
        "dtype": str(arr.dtype),
        "min": float(arr64.min()),
        "max": float(arr64.max()),
        "mean": float(arr64.mean()),
        "sum": float(arr64.sum()),
        "dV_full_cell_ang3": float(dv_full_cell(cell, arr64.shape)),
        "integral_full_cell": float(integrate_density(arr64, cell)),
    }


def print_summary_block(title: str, info: Dict[str, object]) -> None:
    print(title)
    print("-" * 80)
    print(f"shape                    : {info['shape']}")
    print(f"dtype                    : {info['dtype']}")
    print(f"min                      : {info['min']:.10f}")
    print(f"max                      : {info['max']:.10f}")
    print(f"mean                     : {info['mean']:.10f}")
    print(f"sum                      : {info['sum']:.10f}")
    print(f"dV full-cell (A^3)       : {info['dV_full_cell_ang3']:.10f}")
    print(f"integral full-cell       : {info['integral_full_cell']:.10f}")
    print()


def compare_arrays(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        print(f"{label_a} vs {label_b}")
        print("-" * 80)
        print(f"shape mismatch           : {a.shape} vs {b.shape}")
        print()
        return

    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    diff = a64 - b64
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))

    print(f"{label_a} vs {label_b}")
    print("-" * 80)
    print(f"shape                    : {a.shape}")
    print(f"MAE                      : {mae:.10e}")
    print(f"RMSE                     : {rmse:.10e}")
    print(f"max |diff|               : {max_abs:.10e}")
    print()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_qm9_atoms(db_path: str, mol_key: str) -> Tuple[Atoms, str]:
    with h5py.File(db_path, "r") as db:
        if mol_key not in db:
            raise KeyError(f"Molecule key {mol_key!r} not found in {db_path}")
        grp = db[mol_key]
        positions = grp["positions"][:]
        numbers = grp["numbers"][:]
        atoms = Atoms(numbers=numbers, positions=positions)
        formula = atoms.get_chemical_formula()
    return atoms, formula


def load_stored_group(stored_path: str, mol_key: str):
    with h5py.File(stored_path, "r") as f:
        if mol_key not in f:
            raise KeyError(
                f"Molecule key {mol_key!r} not found in stored dataset {stored_path}"
            )
        grp = f[mol_key]
        n_r = grp["n_r"][:]
        v_ext = grp["v_ext"][:]
        attrs = {k: grp.attrs[k] for k in grp.attrs.keys()}
    return n_r, v_ext, attrs


def save_output_h5(
    out_path: str,
    mol_key: str,
    formula: str,
    atoms: Atoms,
    energy_hartree: float,
    arrays: Dict[str, np.ndarray],
    attrs_extra: Dict[str, object],
) -> None:
    ensure_parent(out_path)
    with h5py.File(out_path, "w") as f:
        f.attrs["description"] = "Focused GPAW density-mode verification output"
        f.attrs["molecule_key"] = mol_key
        f.attrs["formula"] = formula
        f.attrs["cell_angstrom"] = atoms.cell.array
        f.attrs["total_energy_hartree"] = energy_hartree

        for key, value in attrs_extra.items():
            f.attrs[key] = value

        for name, arr in arrays.items():
            if np.issubdtype(arr.dtype, np.floating):
                data = np.asarray(arr, dtype=np.float32)
            else:
                data = arr
            f.create_dataset(
                name,
                data=data,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )


def main() -> None:
    args = parse_args()
    t0 = time.time()

    ensure_parent(args.out_path)
    os.makedirs(args.log_dir, exist_ok=True)

    atoms, formula = load_qm9_atoms(args.db_path, args.mol_key)
    total_e = electron_count_from_formula(formula, TOTAL_ELECTRONS)
    valence_e = electron_count_from_formula(formula, VALENCE_ELECTRONS)

    print("=" * 80)
    print("GPAW CH4 / QM9 DENSITY MODE TEST")
    print("=" * 80)
    print(f"db_path                  : {args.db_path}")
    print(f"stored_path              : {args.stored_path}")
    print(f"mol_key                  : {args.mol_key}")
    print(f"formula                  : {formula}")
    print(f"total electrons          : {total_e}")
    print(f"valence electrons        : {valence_e}")
    print()

    padding_angstrom = args.padding_bohr * Bohr
    atoms.center(vacuum=padding_angstrom)

    print("Centered atoms")
    print("-" * 80)
    print(f"padding_bohr             : {args.padding_bohr}")
    print(f"padding_angstrom         : {padding_angstrom:.10f}")
    print(f"cell_angstrom            :\n{atoms.cell.array}")
    print(f"cell volume (A^3)        : {cell_volume_ang3(atoms.cell.array):.10f}")
    print()

    gpaw_log = os.path.join(args.log_dir, f"log_{args.mol_key}.txt")
    calc = GPAW(h=args.h, mode=args.mode, xc=args.xc, txt=gpaw_log, spinpol=False)
    atoms.calc = calc

    try:
        e_pot = atoms.get_potential_energy()
        e_ref = calc.get_reference_energy()
        e_total_hartree = float((e_pot + e_ref) / Hartree)

        # Pseudo-density
        n_pseudo_pad = np.asarray(calc.get_pseudo_density(pad=True), dtype=np.float64)
        n_pseudo_nopad = np.asarray(
            calc.get_pseudo_density(pad=False), dtype=np.float64
        )

        # All-electron density
        n_ae_pad = np.asarray(
            calc.get_all_electron_density(gridrefinement=args.gridrefinement, pad=True),
            dtype=np.float64,
        )
        n_ae_nopad = np.asarray(
            calc.get_all_electron_density(
                gridrefinement=args.gridrefinement, pad=False
            ),
            dtype=np.float64,
        )

        # Potential (for completeness; not a density)
        v_ext = np.asarray(calc.get_electrostatic_potential(), dtype=np.float64)

    finally:
        atoms.calc = None

    print("Energy")
    print("-" * 80)
    print(f"total_energy_hartree     : {e_total_hartree:.10f}")
    print(f"gpaw log                 : {gpaw_log}")
    print()

    pseudo_pad_info = summarize_array("pseudo_pad", n_pseudo_pad, atoms.cell.array)
    pseudo_nopad_info = summarize_array(
        "pseudo_nopad", n_pseudo_nopad, atoms.cell.array
    )
    ae_pad_info = summarize_array("all_electron_pad", n_ae_pad, atoms.cell.array)
    ae_nopad_info = summarize_array("all_electron_nopad", n_ae_nopad, atoms.cell.array)
    vext_info = summarize_array("v_ext", v_ext, atoms.cell.array)

    print_summary_block("Pseudo density (pad=True)", pseudo_pad_info)
    print_summary_block("Pseudo density (pad=False)", pseudo_nopad_info)
    print_summary_block(
        f"All-electron density (gridrefinement={args.gridrefinement}, pad=True)",
        ae_pad_info,
    )
    print_summary_block(
        f"All-electron density (gridrefinement={args.gridrefinement}, pad=False)",
        ae_nopad_info,
    )
    print_summary_block("Electrostatic potential (NOT a density)", vext_info)

    print("Electron-count interpretation")
    print("-" * 80)
    print(
        f"pseudo pad=True vs valence    : "
        f"{abs(pseudo_pad_info['integral_full_cell'] - valence_e):.10f}"
    )
    print(
        f"pseudo pad=True vs total      : "
        f"{abs(pseudo_pad_info['integral_full_cell'] - total_e):.10f}"
    )
    print(
        f"pseudo pad=False vs valence   : "
        f"{abs(pseudo_nopad_info['integral_full_cell'] - valence_e):.10f}"
    )
    print(
        f"pseudo pad=False vs total     : "
        f"{abs(pseudo_nopad_info['integral_full_cell'] - total_e):.10f}"
    )
    print(
        f"all-electron pad=True vs total: "
        f"{abs(ae_pad_info['integral_full_cell'] - total_e):.10f}"
    )
    print(
        f"all-electron pad=False vs total: "
        f"{abs(ae_nopad_info['integral_full_cell'] - total_e):.10f}"
    )
    print()
    print("Interpretation:")
    print(
        "  - Pseudo density is expected to behave like a pseudo/valence-like density."
    )
    print(
        "  - All-electron density is the quantity to compare against total electrons."
    )
    print("  - v_ext is a potential and must NOT be added to n(r).")
    print()

    # Compare fresh pseudo-density against float32-cast pseudo-density
    n_pseudo_pad_f32 = n_pseudo_pad.astype(np.float32).astype(np.float64)
    compare_arrays(
        "fresh pseudo pad=True",
        n_pseudo_pad,
        "fresh pseudo pad=True cast fp32",
        n_pseudo_pad_f32,
    )

    # Optional compare against stored merged dataset
    stored_arrays = {}
    if not args.skip_stored_compare and os.path.exists(args.stored_path):
        stored_n_r, stored_v_ext, stored_attrs = load_stored_group(
            args.stored_path, args.mol_key
        )

        print("Stored dataset comparison")
        print("-" * 80)
        print(f"stored formula            : {stored_formula}")
        print(f"stored n_r shape          : {stored_n_r.shape}")
        print(f"stored v_ext shape        : {stored_v_ext.shape}")
        print(
            f"stored n_r integral       : "
            f"{integrate_density(stored_n_r, atoms.cell.array):.10f}"
        )
        print()

        compare_arrays("fresh pseudo pad=True", n_pseudo_pad, "stored n_r", stored_n_r)
        stored_arrays["stored_n_r"] = np.asarray(stored_n_r, dtype=np.float32)
        stored_arrays["stored_v_ext"] = np.asarray(stored_v_ext, dtype=np.float32)
    else:
        print("Stored dataset comparison")
        print("-" * 80)
        print("Skipped.")
        print()

    arrays_to_save = {
        "pseudo_density_pad_true": np.asarray(n_pseudo_pad, dtype=np.float32),
        "pseudo_density_pad_false": np.asarray(n_pseudo_nopad, dtype=np.float32),
        "all_electron_density_pad_true": np.asarray(n_ae_pad, dtype=np.float32),
        "all_electron_density_pad_false": np.asarray(n_ae_nopad, dtype=np.float32),
        "electrostatic_potential": np.asarray(v_ext, dtype=np.float32),
        **stored_arrays,
    }

    attrs_extra = {
        "grid_spacing_angstrom": args.h,
        "padding_bohr": args.padding_bohr,
        "xc": args.xc,
        "mode": args.mode,
        "gridrefinement": args.gridrefinement,
        "expected_total_electrons": total_e,
        "expected_valence_electrons": valence_e,
        "pseudo_pad_true_integral": pseudo_pad_info["integral_full_cell"],
        "pseudo_pad_false_integral": pseudo_nopad_info["integral_full_cell"],
        "all_electron_pad_true_integral": ae_pad_info["integral_full_cell"],
        "all_electron_pad_false_integral": ae_nopad_info["integral_full_cell"],
        "elapsed_seconds": time.time() - t0,
    }

    save_output_h5(
        out_path=args.out_path,
        mol_key=args.mol_key,
        formula=formula,
        atoms=atoms,
        energy_hartree=e_total_hartree,
        arrays=arrays_to_save,
        attrs_extra=attrs_extra,
    )

    print("Saved output")
    print("-" * 80)
    print(f"out_path                 : {args.out_path}")
    print(f"elapsed_seconds          : {time.time() - t0:.3f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
