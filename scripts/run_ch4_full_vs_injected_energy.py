#!/usr/bin/env python3
"""Run CH4 full-SCF vs injected single-shot energy comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full SCF for CH4, inject saved pseudo-density into a fresh GPAW "
            "calculator, and compare energy components."
        )
    )
    parser.add_argument(
        "--db-path",
        default="/home/achmadjae/gpaw-qm9/dataset/raw/qm9_smallest_20k.h5",
        help="Input raw QM9 HDF5 file.",
    )
    parser.add_argument(
        "--mol-key",
        default="dsgdb9nsd_000001",
        help="Molecule key in HDF5 (default CH4).",
    )
    parser.add_argument("--h", type=float, default=0.2, help="GPAW grid spacing (Angstrom).")
    parser.add_argument("--xc", default="PBE", help="XC functional.")
    parser.add_argument("--mode", default="fd", help="GPAW mode.")
    parser.add_argument("--setups", default="hgh", help="PAW setups.")
    parser.add_argument(
        "--padding-bohr",
        type=float,
        default=4.0,
        help="Vacuum padding in Bohr.",
    )
    parser.add_argument(
        "--diag-iters",
        type=int,
        default=20,
        help="Max eigensolver iterations for fixed-density single-shot.",
    )
    parser.add_argument(
        "--eig-tol",
        type=float,
        default=1e-8,
        help="Eigenvalue stability tolerance (eV).",
    )
    parser.add_argument(
        "--out-json",
        default="/home/achmadjae/gpaw-qm9/logs/ch4_full_vs_injected_energy.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-npy",
        default="/home/achmadjae/gpaw-qm9/logs/ch4_full_scf_n_pseudo.npy",
        help="Output NPY path for full-SCF pseudo-density.",
    )
    return parser.parse_args()


def axis_slice(src_n: int, dst_n: int) -> slice:
    if src_n == dst_n:
        return slice(0, dst_n)
    if src_n == dst_n + 1:
        return slice(1, 1 + dst_n)
    if src_n > dst_n:
        start = (src_n - dst_n) // 2
        return slice(start, start + dst_n)
    raise ValueError(f"Cannot map source axis {src_n} to destination axis {dst_n}")


def inject_pseudo_density(calc: GPAW, n_pseudo_h5: np.ndarray) -> None:
    nx, ny, nz = calc.density.nt_sG.shape[1:]
    sx = axis_slice(int(n_pseudo_h5.shape[0]), nx)
    sy = axis_slice(int(n_pseudo_h5.shape[1]), ny)
    sz = axis_slice(int(n_pseudo_h5.shape[2]), nz)
    n_internal = n_pseudo_h5[sx, sy, sz] * (Bohr ** 3)
    if n_internal.shape != (nx, ny, nz):
        raise ValueError(f"Mapped density shape {n_internal.shape} != target {(nx, ny, nz)}")
    calc.density.nt_sG[0, :] = n_internal
    calc.density.interpolate_pseudo_density()
    calc.density.calculate_pseudo_charge()
    calc.hamiltonian.update(calc.density)


def run_fixed_density_eigs(calc: GPAW, max_iters: int, stability_tol: float) -> None:
    prev = None
    for _ in range(max_iters):
        calc.wfs.eigensolver.iterate(calc.hamiltonian, calc.wfs)
        eig = calc.get_eigenvalues(spin=0)
        if prev is not None and np.max(np.abs(eig - prev)) < stability_tol:
            break
        prev = eig.copy()


def energy_dict(calc: GPAW) -> dict[str, float | None]:
    h = calc.hamiltonian
    out = {
        "e_total_free_Ha": float(h.e_total_free) if h.e_total_free is not None else None,
        "e_total_extrapolated_Ha": (
            float(h.e_total_extrapolated) if h.e_total_extrapolated is not None else None
        ),
        "e_kinetic_Ha": float(h.e_kinetic) if h.e_kinetic is not None else None,
        "e_kinetic0_Ha": float(h.e_kinetic0) if getattr(h, "e_kinetic0", None) is not None else None,
        "e_coulomb_Ha": float(h.e_coulomb) if h.e_coulomb is not None else None,
        "e_xc_Ha": float(h.e_xc) if h.e_xc is not None else None,
        "e_zero_Ha": float(h.e_zero) if h.e_zero is not None else None,
        "e_external_Ha": float(h.e_external) if h.e_external is not None else None,
        "e_band_Ha": float(h.e_band) if h.e_band is not None else None,
    }
    for key, value in list(out.items()):
        out[key.replace("_Ha", "_eV")] = None if value is None else float(value * Hartree)
    return out


def main() -> None:
    args = parse_args()

    out_json = Path(args.out_json)
    out_npy = Path(args.out_npy)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.db_path, "r") as db:
        if args.mol_key not in db:
            raise KeyError(f"{args.mol_key!r} not found in {args.db_path}")
        group = db[args.mol_key]
        positions = np.asarray(group["positions"][:], dtype=np.float64)
        numbers = np.asarray(group["numbers"][:], dtype=np.int32)

    atoms_full = Atoms(numbers=numbers, positions=positions)
    atoms_full.center(vacuum=args.padding_bohr * Bohr)
    calc_full = GPAW(
        h=args.h,
        mode=args.mode,
        xc=args.xc,
        setups=args.setups,
        txt=None,
        spinpol=False,
    )
    atoms_full.calc = calc_full
    full_energy_eV = float(atoms_full.get_potential_energy())
    full_ref_eV = float(calc_full.get_reference_energy())
    n_pseudo = np.asarray(calc_full.get_pseudo_density(pad=True), dtype=np.float64)
    np.save(out_npy, n_pseudo)

    full = energy_dict(calc_full)
    full["potential_energy_eV"] = full_energy_eV
    full["reference_energy_eV"] = full_ref_eV
    full["potential_plus_reference_eV"] = full_energy_eV + full_ref_eV

    atoms_inj = Atoms(numbers=numbers, positions=positions)
    atoms_inj.center(vacuum=args.padding_bohr * Bohr)
    calc_inj = GPAW(
        h=args.h,
        mode=args.mode,
        xc=args.xc,
        setups=args.setups,
        txt=None,
        spinpol=False,
        nbands=int(calc_full.get_number_of_bands()),
    )
    atoms_inj.calc = calc_inj
    calc_inj.initialize(atoms_inj)
    calc_inj.set_positions(atoms_inj)
    inject_pseudo_density(calc_inj, n_pseudo)
    run_fixed_density_eigs(calc_inj, max_iters=args.diag_iters, stability_tol=args.eig_tol)
    e_entropy = calc_inj.wfs.calculate_occupation_numbers()
    calc_inj.hamiltonian.get_energy(e_entropy, calc_inj.wfs)
    injected = energy_dict(calc_inj)

    comparison: dict[str, dict[str, float]] = {}
    for key in sorted(set(full) & set(injected)):
        v_full = full[key]
        v_inj = injected[key]
        if isinstance(v_full, float) and isinstance(v_inj, float):
            comparison[key] = {
                "full_scf": v_full,
                "injected_single_shot": v_inj,
                "delta_injected_minus_full": v_inj - v_full,
                "abs_delta": abs(v_inj - v_full),
            }

    report = {
        "db_path": args.db_path,
        "molecule_key": args.mol_key,
        "numbers": numbers.tolist(),
        "settings": {
            "h": args.h,
            "mode": args.mode,
            "xc": args.xc,
            "setups": args.setups,
            "padding_bohr": args.padding_bohr,
            "diag_iters": args.diag_iters,
            "eig_tol": args.eig_tol,
        },
        "artifacts": {
            "full_scf_n_pseudo_npy": str(out_npy),
            "report_json": str(out_json),
        },
        "full_scf_energy_components": full,
        "injected_energy_components": injected,
        "comparison": comparison,
    }
    out_json.write_text(json.dumps(report, indent=2))

    print("Saved:", out_json)
    print("Saved:", out_npy)
    for key in [
        "e_band_eV",
        "e_kinetic_eV",
        "e_total_free_eV",
        "e_coulomb_eV",
        "e_xc_eV",
        "e_zero_eV",
    ]:
        if key in comparison:
            d = comparison[key]
            print(
                f"{key}: full={d['full_scf']:.12f} "
                f"inj={d['injected_single_shot']:.12f} "
                f"delta={d['delta_injected_minus_full']:+.3e}"
            )


if __name__ == "__main__":
    main()
