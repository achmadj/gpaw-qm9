#!/usr/bin/env python3
"""
Test GPAW PAW consistency under rotation for gdb_1006 (C6H4O).
Setup: LDA, h=0.2, fd mode, padding=4.0 Bohr.

Tests 6 cases:
1. Full SCF on original geometry.
2. Single-shot injection of original nt_sG[0] + D_asp into original geometry.
3. Full SCF on 90 degrees rotated geometry (around y).
4. Single-shot injection of 90 degrees rotated nt_sG[0] + D_asp into 90 degrees rotated geometry.
5. Full SCF on 180 degrees rotated geometry (around y).
6. Single-shot injection of 180 degrees rotated nt_sG[0] + D_asp into 180 degrees rotated geometry.

We extract the unpadded pseudo-density directly from calc.density.nt_sG[0]
(in units e/Bohr^3) to avoid padding mismatches when rotating.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import h5py
from ase import Atoms
from ase.units import Bohr, Hartree
from scipy.ndimage import rotate
from gpaw import GPAW
from gpaw.rotation import rotation as gpaw_rotation
from gpaw.utilities import pack_density, unpack_density


# --- PREVENT SLURM MPI CONTEXT LEAKAGE ---
for _k in list(os.environ.keys()):
    if _k.startswith(("PMIX_", "PMI_", "SLURM_MPI_", "SLURM_PMIX")):
        os.environ.pop(_k, None)


def extract_D_asp(calc: GPAW) -> dict[int, np.ndarray]:
    """Extract a deep copy of D_asp from a calculator."""
    D_asp = {}
    for a, D_sp in calc.density.D_asp.items():
        D_asp[int(a)] = np.array(D_sp, dtype=np.float64)
    return D_asp


def rotate_D_asp(D_asp_in: dict[int, np.ndarray], setups, U_vv: np.ndarray) -> dict[int, np.ndarray]:
    """Rotate atomic density matrices by global Cartesian rotation U_vv."""
    D_asp_out = {}
    for a, D_sp in D_asp_in.items():
        setup = setups[a]
        nspins = D_sp.shape[0]
        ni = setup.ni
        R_ii = np.zeros((ni, ni))
        i1 = 0
        for l in setup.l_j:
            i2 = i1 + 2 * l + 1
            R_mm = gpaw_rotation(l, U_vv)
            R_ii[i1:i2, i1:i2] = R_mm
            i1 = i2

        D_sp_rot = np.empty_like(D_sp)
        for s in range(nspins):
            D_ii = unpack_density(D_sp[s])
            D_ii_rot = R_ii @ D_ii @ R_ii.T
            D_sp_rot[s] = pack_density(D_ii_rot)
        D_asp_out[a] = D_sp_rot
    return D_asp_out


def inject_pseudo_density_and_dasp(calc: GPAW, n_pseudo: np.ndarray, D_asp_dict: dict[int, np.ndarray]) -> None:
    """Inject n_pseudo (unpadded, e/Bohr^3) and D_asp into a fresh calculator."""
    target_shape = calc.density.nt_sG[0].shape
    if n_pseudo.shape != target_shape:
        raise ValueError(
            f"Density shape mismatch: injected {n_pseudo.shape} vs target {target_shape}"
        )
    calc.density.nt_sG[0, :] = n_pseudo

    for a, D_sp in D_asp_dict.items():
        calc.density.D_asp[int(a)] = D_sp.copy()

    calc.density.interpolate_pseudo_density()
    calc.density.calculate_pseudo_charge()
    calc.hamiltonian.update(calc.density)


def run_fixed_density_eigs(calc: GPAW, max_iters: int = 20, stability_tol: float = 1e-8) -> None:
    """Run fixed-density eigensolver without SCF loop."""
    prev = None
    for _ in range(max_iters):
        calc.wfs.eigensolver.iterate(calc.hamiltonian, calc.wfs)
        eig = calc.get_eigenvalues(spin=0)
        if prev is not None and np.max(np.abs(eig - prev)) < stability_tol:
            break
        prev = eig.copy()


def energy_dict(calc: GPAW) -> dict:
    h = calc.hamiltonian
    out = {}
    for attr in ["e_total_free", "e_total_extrapolated", "e_kinetic", "e_kinetic0",
                 "e_coulomb", "e_xc", "e_zero", "e_external", "e_band"]:
        val = getattr(h, attr, None)
        out[f"{attr}_Ha"] = float(val) if val is not None else None
        out[f"{attr}_eV"] = float(val * Hartree) if val is not None else None
    return out


def eigenvalue_dict(calc: GPAW, n_occupied: int) -> dict:
    evals = calc.get_eigenvalues(spin=0)
    homo = evals[n_occupied - 1]
    lumo = evals[n_occupied] if n_occupied < len(evals) else float('nan')
    gap = lumo - homo if n_occupied < len(evals) else float('nan')
    return {
        "eigenvalues_eV": evals.tolist(),
        "homo_eV": float(homo),
        "lumo_eV": float(lumo),
        "gap_eV": float(gap),
    }


def run_full_scf(atoms: Atoms, calc_kwargs: dict, label: str):
    print(f"\n{'='*60}")
    print(f"Running FULL SCF: {label}")
    print(f"{'='*60}")

    calc = GPAW(**calc_kwargs)
    atoms_c = atoms.copy()
    atoms_c.calc = calc

    t0 = time.time()
    e_pot = atoms_c.get_potential_energy()
    elapsed = time.time() - t0

    n_electrons = int(sum(atoms.numbers))
    n_occupied = n_electrons // 2

    # Extract unpadded pseudo-density directly from nt_sG (e/Bohr^3)
    nt_sG = np.array(calc.density.nt_sG[0], dtype=np.float64)
    D_asp = extract_D_asp(calc)

    result = {
        "label": label,
        "type": "full_scf",
        "potential_energy_eV": float(e_pot),
        "n_scf_iterations": calc.scf.niter,
        "elapsed_seconds": elapsed,
        "grid_shape": list(nt_sG.shape),
        **energy_dict(calc),
        **eigenvalue_dict(calc, n_occupied),
    }

    print(f"  Energy: {result['e_total_free_eV']:.6f} eV  ({calc.scf.niter} iterations)")
    print(f"  HOMO: {result['homo_eV']:.4f} eV  LUMO: {result['lumo_eV']:.4f} eV  gap: {result['gap_eV']:.4f} eV")
    print(f"  Grid: {nt_sG.shape}")

    atoms_c.calc = None
    return result, nt_sG, D_asp, calc


def run_injected(atoms: Atoms, calc_kwargs: dict, nt_sG: np.ndarray, D_asp: dict, label: str, nbands: int):
    print(f"\n{'='*60}")
    print(f"Running INJECTED SINGLE-SHOT: {label}")
    print(f"{'='*60}")

    inj_kwargs = calc_kwargs.copy()
    inj_kwargs["nbands"] = nbands

    calc = GPAW(**inj_kwargs)
    atoms_c = atoms.copy()
    atoms_c.calc = calc

    calc.initialize(atoms_c)
    calc.set_positions(atoms_c)

    t0 = time.time()
    inject_pseudo_density_and_dasp(calc, nt_sG, D_asp)
    run_fixed_density_eigs(calc, max_iters=20, stability_tol=1e-8)
    e_entropy = calc.wfs.calculate_occupation_numbers()
    calc.hamiltonian.get_energy(e_entropy, calc.wfs)
    elapsed = time.time() - t0

    n_electrons = int(sum(atoms.numbers))
    n_occupied = n_electrons // 2

    result = {
        "label": label,
        "type": "injected_single_shot",
        "elapsed_seconds": elapsed,
        "grid_shape": list(calc.density.nt_sG[0].shape),
        **energy_dict(calc),
        **eigenvalue_dict(calc, n_occupied),
    }

    print(f"  Energy: {result['e_total_free_eV']:.6f} eV")
    print(f"  HOMO: {result['homo_eV']:.4f} eV  LUMO: {result['lumo_eV']:.4f} eV  gap: {result['gap_eV']:.4f} eV")

    atoms_c.calc = None
    return result


def compare_pair(ref: dict, inj: dict) -> dict:
    comp = {}
    for key in ["homo_eV", "lumo_eV", "gap_eV", "e_total_free_eV", "e_band_eV"]:
        v_ref = ref.get(key)
        v_inj = inj.get(key)
        if v_ref is not None and v_inj is not None:
            comp[f"delta_{key}"] = v_inj - v_ref
            comp[f"abs_delta_{key}"] = abs(v_inj - v_ref)
    if ref.get("eigenvalues_eV") and inj.get("eigenvalues_eV"):
        ev_ref = np.array(ref["eigenvalues_eV"])
        ev_inj = np.array(inj["eigenvalues_eV"])
        n = min(len(ev_ref), len(ev_inj))
        comp["max_eigenvalue_error_eV"] = float(np.max(np.abs(ev_ref[:n] - ev_inj[:n])))
    return comp


def print_summary(report: dict) -> None:
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    cases = report["cases"]
    comps = report["comparisons"]

    print("\n--- Full SCF vs Injected (Single Shot) ---")
    for label in ["original", "rotated_90_y", "rotated_180_y"]:
        full = cases[f"{label}_full_scf"]
        inj = cases[f"{label}_injected"]
        c = comps[label]
        print(f"\n{label}:")
        print(f"  Full SCF:  E={full['e_total_free_eV']:.6f} eV  HOMO={full['homo_eV']:.4f}  LUMO={full['lumo_eV']:.4f}  gap={full['gap_eV']:.4f}")
        print(f"  Injected:  E={inj['e_total_free_eV']:.6f} eV  HOMO={inj['homo_eV']:.4f}  LUMO={inj['lumo_eV']:.4f}  gap={inj['gap_eV']:.4f}")
        print(f"  dE={c.get('delta_e_total_free_eV', float('nan')):+.3e}  dHOMO={c.get('delta_homo_eV', float('nan')):+.3e}  max|de|={c.get('max_eigenvalue_error_eV', float('nan')):.3e} eV")

    print("\n--- Rotation Invariance (Full SCF) ---")
    c90 = comps["rotation_invariance_full_scf"]
    c180 = comps["rotation_invariance_full_scf_180"]
    print(f"Original vs 90:   dE={c90.get('delta_e_total_free_eV', float('nan')):+.3e}  dHOMO={c90.get('delta_homo_eV', float('nan')):+.3e}")
    print(f"Original vs 180:  dE={c180.get('delta_e_total_free_eV', float('nan')):+.3e}  dHOMO={c180.get('delta_homo_eV', float('nan')):+.3e}")


def main():
    db_path = "/home/achmadjae/gpaw-qm9/dataset/qm9_1000_phase_a.h5"
    mol_key = "gdb_1006"

    out_dir = Path("/home/achmadjae/gpaw-qm9/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "gdb1006_lda_rotation_report.json"

    print(f"Loading {mol_key} from {db_path}...")
    with h5py.File(db_path, "r") as f:
        grp = f[mol_key]
        positions = grp["positions"][:]
        numbers = grp["numbers"][:]

    atoms0 = Atoms(numbers=numbers, positions=positions)
    atoms0.center(vacuum=4.0 * Bohr)

    calc_kwargs = dict(h=0.2, mode="fd", xc="LDA", spinpol=False, txt=None)

    # --- Case 1 & 2: Original ---
    res0, nt_sG0, D_asp0, calc0 = run_full_scf(atoms0, calc_kwargs, "original")
    inj0 = run_injected(atoms0, calc_kwargs, nt_sG0, D_asp0, "original", nbands=calc0.get_number_of_bands())

    # --- Case 3 & 4: 90 degrees rotation around y ---
    atoms90 = atoms0.copy()
    atoms90.rotate(90, 'y')
    atoms90.center(vacuum=4.0 * Bohr)

    res90, nt_sG90_ref, D_asp90_ref, calc90 = run_full_scf(atoms90, calc_kwargs, "rotated_90_y")

    # scipy.ndimage.rotate uses the opposite handedness compared to ASE for axes=(0,2).
    # Empirically, ASE rotate(+90, 'y') matches scipy rotate(-90, axes=(0,2)).
    nt_sG90_rot = rotate(nt_sG0, -90, axes=(0, 2), reshape=True, order=1, prefilter=False, mode='constant', cval=0.0)
    # ASE rotates column vectors: r' = R @ r.
    # GPAW rotation(l, U_vv) uses row-vector convention: r'^T = r^T @ U_vv,
    # so U_vv must be R.T.
    R_90 = np.array([[0, 0, 1],
                     [0, 1, 0],
                     [-1, 0, 0]], dtype=float)
    U_90 = R_90.T
    D_asp90_rot = rotate_D_asp(D_asp0, calc0.wfs.setups, U_90)

    inj90 = run_injected(atoms90, calc_kwargs, nt_sG90_rot, D_asp90_rot, "rotated_90_y", nbands=calc90.get_number_of_bands())

    # --- Case 5 & 6: 180 degrees rotation around y ---
    atoms180 = atoms0.copy()
    atoms180.rotate(180, 'y')
    atoms180.center(vacuum=4.0 * Bohr)

    res180, nt_sG180_ref, D_asp180_ref, calc180 = run_full_scf(atoms180, calc_kwargs, "rotated_180_y")

    nt_sG180_rot = rotate(nt_sG0, 180, axes=(0, 2), reshape=True, order=1, prefilter=False, mode='constant', cval=0.0)
    U_180 = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]], dtype=float)
    D_asp180_rot = rotate_D_asp(D_asp0, calc0.wfs.setups, U_180)

    inj180 = run_injected(atoms180, calc_kwargs, nt_sG180_rot, D_asp180_rot, "rotated_180_y", nbands=calc180.get_number_of_bands())

    # --- Compile report ---
    report = {
        "molecule": mol_key,
        "formula": atoms0.get_chemical_formula(),
        "setup": calc_kwargs,
        "cases": {
            "original_full_scf": res0,
            "original_injected": inj0,
            "rotated_90_y_full_scf": res90,
            "rotated_90_y_injected": inj90,
            "rotated_180_y_full_scf": res180,
            "rotated_180_y_injected": inj180,
        },
        "comparisons": {
            "original": compare_pair(res0, inj0),
            "rotated_90_y": compare_pair(res90, inj90),
            "rotated_180_y": compare_pair(res180, inj180),
            "rotation_invariance_full_scf": compare_pair(res0, res90),
            "rotation_invariance_full_scf_180": compare_pair(res0, res180),
        }
    }

    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_json}")

    print_summary(report)


if __name__ == "__main__":
    main()
