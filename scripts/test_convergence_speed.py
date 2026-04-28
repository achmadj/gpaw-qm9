#!/usr/bin/env python3
"""
Test convergence speed: injected converged density vs default initial guess.
Run N SCF steps and track eigenvalue convergence at each step.
"""

import numpy as np
import h5py
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW

HARTREE_TO_EV = 27.211386245988


def run_with_maxiter(atoms, maxiter, nt_sG_inject=None, D_asp_inject=None):
    """Run GPAW with given maxiter, optionally injecting density."""
    calc = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False,
                maxiter=maxiter,
                convergence={'energy': 1e-8, 'density': 1e-8})
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc

    if nt_sG_inject is not None:
        calc.initialize(atoms_copy)
        calc.set_positions(atoms_copy)
        calc.density.nt_sG[:] = nt_sG_inject
        if D_asp_inject is not None:
            for a, D_sp in D_asp_inject.items():
                calc.density.D_asp[a][:] = D_sp
        calc.density.calculate_pseudo_charge()

    try:
        e = atoms_copy.get_potential_energy()
    except:
        e = float('nan')

    try:
        evals = calc.get_eigenvalues(spin=0)
    except:
        evals = None

    niter = calc.scf.niter if hasattr(calc.scf, 'niter') else maxiter
    atoms_copy.calc = None
    return e, evals, niter


def main():
    db_path = '/home/achmadjae/gpaw-qm9/dataset/raw/qm9_smallest_20k.h5'
    with h5py.File(db_path, 'r') as db:
        grp = db['dsgdb9nsd_000001']
        positions = grp['positions'][:]
        numbers = grp['numbers'][:]

    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.center(vacuum=4.0 * Bohr)
    n_occupied = sum(numbers) // 2

    # Full convergence (ground truth)
    print("Running full SCF convergence...")
    calc_ref = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False,
                    convergence={'energy': 1e-8, 'density': 1e-8})
    atoms_ref = atoms.copy()
    atoms_ref.calc = calc_ref
    e_ref = atoms_ref.get_potential_energy()
    evals_ref = calc_ref.get_eigenvalues(spin=0)
    homo_ref = evals_ref[n_occupied - 1]
    lumo_ref = evals_ref[n_occupied]
    gap_ref = lumo_ref - homo_ref
    niter_ref = calc_ref.scf.niter

    # Save converged density
    nt_sG_conv = calc_ref.density.nt_sG.copy()
    D_asp_conv = {}
    for a, D_sp in calc_ref.density.D_asp.items():
        D_asp_conv[a] = D_sp.copy()
    atoms_ref.calc = None

    print(f"Reference: E={e_ref:.6f} eV, HOMO={homo_ref:.4f}, LUMO={lumo_ref:.4f}, Gap={gap_ref:.4f} eV, {niter_ref} iter\n")

    # Test convergence at 1, 2, 3, 5, 10, 20 iterations
    test_iters = [1, 2, 3, 5, 10, 15, 20]

    print(f"{'Maxiter':<10} {'Mode':<12} {'HOMO err':<12} {'LUMO err':<12} {'Gap err':<12} {'E err (eV)':<14} {'Niter':<8}")
    print("-" * 80)

    for maxiter in test_iters:
        # With injected converged density
        e_inj, evals_inj, niter_inj = run_with_maxiter(
            atoms, maxiter, nt_sG_conv, D_asp_conv)

        if evals_inj is not None:
            homo_inj = evals_inj[n_occupied - 1]
            lumo_inj = evals_inj[n_occupied]
            gap_inj = lumo_inj - homo_inj
            print(f"{maxiter:<10} {'injected':<12} {homo_inj-homo_ref:<12.4f} {lumo_inj-lumo_ref:<12.4f} {gap_inj-gap_ref:<12.4f} {e_inj-e_ref:<14.6f} {niter_inj:<8}")
        else:
            print(f"{maxiter:<10} {'injected':<12} {'FAIL':<12}")

        # With default initial guess
        e_def, evals_def, niter_def = run_with_maxiter(atoms, maxiter)

        if evals_def is not None:
            homo_def = evals_def[n_occupied - 1]
            lumo_def = evals_def[n_occupied]
            gap_def = lumo_def - homo_def
            print(f"{maxiter:<10} {'default':<12} {homo_def-homo_ref:<12.4f} {lumo_def-lumo_ref:<12.4f} {gap_def-gap_ref:<12.4f} {e_def-e_ref:<14.6f} {niter_def:<8}")
        else:
            print(f"{maxiter:<10} {'default':<12} {'FAIL':<12}")

        print()


if __name__ == '__main__':
    main()
