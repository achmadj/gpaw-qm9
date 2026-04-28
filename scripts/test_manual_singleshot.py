#!/usr/bin/env python3
"""
DEFINITIVE TEST: Manual single-shot KS diagonalization from pseudo-density.

Uses GPAW's internal APIs directly (no get_potential_energy) to:
1. Perturb converged pseudo-density
2. Rebuild Hamiltonian from perturbed density
3. One diagonalization
4. Compare eigenvalues

This bypasses all the high-level calc machinery that resets density.
"""

import numpy as np
import h5py
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW


def main():
    with h5py.File("/home/achmadjae/gpaw-qm9/dataset/raw/qm9_smallest_20k.h5", "r") as db:
        g = db["dsgdb9nsd_000001"]
        positions = g["positions"][:]
        numbers = g["numbers"][:]

    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.center(vacuum=4.0 * Bohr)
    n_occ = sum(numbers) // 2

    # Full SCF
    print("=== Full SCF ===")
    calc = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False)
    atoms.calc = calc
    e_conv = atoms.get_potential_energy()
    evals_conv = calc.get_eigenvalues(spin=0)
    homo_ref = evals_conv[n_occ - 1]
    lumo_ref = evals_conv[n_occ]
    gap_ref = lumo_ref - homo_ref
    niter = calc.scf.niter
    print(f"  {niter} iter, HOMO={homo_ref:.4f} LUMO={lumo_ref:.4f} gap={gap_ref:.4f}")
    print(f"  evals: {evals_conv}")

    # Save converged density
    nt_conv = calc.density.nt_sG.copy()
    D_conv = {}
    for a in calc.density.D_asp:
        D_conv[a] = calc.density.D_asp[a].copy()

    # Manual single-shot at various noise levels
    print("\n=== Manual single-shot (perturbed density) ===")
    print(f"  {'noise':<8} {'max_err':<12} {'HOMO_err':<12} {'LUMO_err':<12} {'gap_err':<12} {'gap':<10}")
    print(f"  {'-'*66}")

    for noise_pct in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
        # Restore converged density
        calc.density.nt_sG[:] = nt_conv.copy()
        for a in D_conv:
            calc.density.D_asp[a][:] = D_conv[a].copy()

        # Perturb nt_sG
        if noise_pct > 0:
            np.random.seed(42)
            nt = calc.density.nt_sG
            noise = (noise_pct / 100.0) * np.random.randn(*nt.shape) * np.abs(nt)
            calc.density.nt_sG[:] = np.clip(nt + noise, 0, None)

        # Manual pipeline: interpolate → pseudo_charge → hamiltonian → diag
        calc.density.interpolate_pseudo_density()
        calc.density.calculate_pseudo_charge()
        calc.hamiltonian.update(calc.density)
        calc.wfs.eigensolver.iterate(calc.hamiltonian, calc.wfs)
        calc.wfs.calculate_occupation_numbers(calc.scf.fix_fermi_level)

        evals = calc.get_eigenvalues(spin=0)
        homo = evals[n_occ - 1]
        lumo = evals[n_occ]
        gap = lumo - homo
        max_err = np.max(np.abs(evals_conv - evals))

        print(f"  {noise_pct:5.1f}%   {max_err:<12.6f} {homo-homo_ref:<12.6f} {lumo-lumo_ref:<12.6f} {gap-gap_ref:<12.6f} {gap:<10.4f}")

    atoms.calc = None
    print("\nDone.")


if __name__ == '__main__':
    main()
