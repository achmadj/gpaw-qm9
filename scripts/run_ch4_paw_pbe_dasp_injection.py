#!/usr/bin/env python3
"""Run CH4 full-SCF vs injected single-shot (n_pseudo + D_asp) energy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.units import Bohr, Hartree
from gpaw import GPAW


def axis_slice(src_n: int, dst_n: int) -> slice:
    if src_n == dst_n:
        return slice(0, dst_n)
    if src_n == dst_n + 1:
        return slice(1, 1 + dst_n)
    if src_n > dst_n:
        start = (src_n - dst_n) // 2
        return slice(start, start + dst_n)
    raise ValueError(f"Cannot map source axis {src_n} to destination axis {dst_n}")


def inject_pseudo_density_and_dasp(calc: GPAW, n_pseudo: np.ndarray, D_asp_dict: dict[str, np.ndarray]) -> None:
    nx, ny, nz = calc.density.nt_sG.shape[1:]
    sx = axis_slice(int(n_pseudo.shape[0]), nx)
    sy = axis_slice(int(n_pseudo.shape[1]), ny)
    sz = axis_slice(int(n_pseudo.shape[2]), nz)
    n_internal = n_pseudo[sx, sy, sz] * (Bohr ** 3)
    if n_internal.shape != (nx, ny, nz):
        raise ValueError(f"Mapped density shape {n_internal.shape} != target {(nx, ny, nz)}")
    calc.density.nt_sG[0, :] = n_internal
    
    # Inject D_asp
    for a, D_sp in D_asp_dict.items():
        calc.density.D_asp[int(a)] = D_sp.copy()

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
    print("Building CH4...")
    atoms_full = molecule('CH4')
    atoms_full.center(vacuum=4.0 * Bohr)
    
    calc_kwargs = {
        "h": 0.2,
        "mode": "fd",
        "xc": "PBE",
        "txt": None,
        "spinpol": False,
    }

    print("Running full SCF...")
    calc_full = GPAW(**calc_kwargs)
    atoms_full.calc = calc_full
    full_energy_eV = float(atoms_full.get_potential_energy())
    full_ref_eV = float(calc_full.get_reference_energy())
    
    n_pseudo = np.asarray(calc_full.get_pseudo_density(pad=True), dtype=np.float64)
    
    # Extract D_asp
    D_asp_data = {}
    for a, D_sp in calc_full.density.D_asp.items():
        D_asp_data[str(a)] = np.array(D_sp, dtype=np.float64)

    full = energy_dict(calc_full)
    full["potential_energy_eV"] = full_energy_eV
    full["reference_energy_eV"] = full_ref_eV
    full["potential_plus_reference_eV"] = full_energy_eV + full_ref_eV

    print("Setting up injected single-shot...")
    atoms_inj = molecule('CH4')
    atoms_inj.center(vacuum=4.0 * Bohr)
    
    inj_kwargs = calc_kwargs.copy()
    inj_kwargs["nbands"] = int(calc_full.get_number_of_bands())
    calc_inj = GPAW(**inj_kwargs)
    atoms_inj.calc = calc_inj
    
    calc_inj.initialize(atoms_inj)
    calc_inj.set_positions(atoms_inj)
    
    print("Injecting n_pseudo and D_asp...")
    inject_pseudo_density_and_dasp(calc_inj, n_pseudo, D_asp_data)
    
    print("Running fixed-density eigensolver...")
    run_fixed_density_eigs(calc_inj, max_iters=20, stability_tol=1e-8)
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

    print("\n--- RESULTS ---")
    for key in [
        "e_total_free_Ha",
        "e_kinetic_Ha",
        "e_coulomb_Ha",
        "e_xc_Ha",
        "e_zero_Ha",
        "e_band_Ha",
    ]:
        if key in comparison:
            d = comparison[key]
            print(
                f"{key:20s}: full={d['full_scf']:+13.8f}  "
                f"inj={d['injected_single_shot']:+13.8f}  "
                f"delta={d['delta_injected_minus_full']:+.3e}"
            )
            
    # Also check HOMO and LUMO
    full_evals = calc_full.get_eigenvalues(spin=0)
    inj_evals = calc_inj.get_eigenvalues(spin=0)
    
    n_occ = 4 # CH4 valence is 8 e- -> 4 bands
    print("\n--- EIGENVALUES (eV) ---")
    print(f"HOMO full: {full_evals[3]:.6f} | inj: {inj_evals[3]:.6f} | delta: {inj_evals[3]-full_evals[3]:+.3e}")
    print(f"LUMO full: {full_evals[4]:.6f} | inj: {inj_evals[4]:.6f} | delta: {inj_evals[4]-full_evals[4]:+.3e}")


if __name__ == "__main__":
    main()
