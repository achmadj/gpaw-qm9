#!/usr/bin/env python3
"""
End-to-end verification: pseudo-density → inject → single-shot → HOMO/LUMO

Pipeline:
1. Run GPAW SCF for CH4 → converged eigenvalues (ground truth)
2. Extract converged pseudo-density ñ(r) and PAW density matrix D_asp
3. Create a FRESH GPAW calculator
4. Inject the saved density
5. Run exactly 1 SCF iteration (single-shot diagonalization)
6. Compare eigenvalues

If single-shot eigenvalues ≈ converged eigenvalues → pipeline works → retraining viable.
"""

import numpy as np
import copy
import h5py
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW
from gpaw.mpi import world

HARTREE_TO_EV = 27.211386245988


def main():
    db_path = '/home/achmadjae/gpaw-qm9/dataset/raw/qm9_smallest_20k.h5'
    mol_key = 'dsgdb9nsd_000001'  # CH4

    # Load molecule
    with h5py.File(db_path, 'r') as db:
        grp = db[mol_key]
        positions = grp['positions'][:]
        numbers = grp['numbers'][:]

    atoms = Atoms(numbers=numbers, positions=positions)
    formula = atoms.get_chemical_formula()
    num_electrons = sum(numbers)
    n_occupied = num_electrons // 2

    print("=" * 70)
    print("PIPELINE VERIFICATION: pseudo-density → inject → single-shot KS")
    print("=" * 70)
    print(f"Molecule: {formula}, electrons: {num_electrons}")

    padding_angstrom = 4.0 * Bohr
    atoms.center(vacuum=padding_angstrom)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Full SCF convergence → ground truth
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- STEP 1: Full SCF convergence (ground truth) ---")
    calc_ref = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False,
                    convergence={'energy': 1e-6, 'density': 1e-6})
    atoms_ref = atoms.copy()
    atoms_ref.calc = calc_ref
    e_ref = atoms_ref.get_potential_energy()

    homo_ref, lumo_ref = calc_ref.get_homo_lumo()
    evals_ref = calc_ref.get_eigenvalues(spin=0)
    n_scf_ref = calc_ref.scf.niter

    print(f"  Converged in {n_scf_ref} SCF iterations")
    print(f"  Total energy: {e_ref:.6f} eV")
    print(f"  HOMO: {homo_ref:.4f} eV, LUMO: {lumo_ref:.4f} eV")
    print(f"  Gap: {lumo_ref - homo_ref:.4f} eV")
    print(f"  Eigenvalues: {evals_ref}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Extract converged pseudo-density
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- STEP 2: Extract converged pseudo-density ---")

    # The key internal quantities:
    # calc.density.nt_sG  — smooth pseudo-density on coarse grid (nspins, nx, ny, nz)
    # calc.density.D_asp  — PAW density matrices (atom-centered)
    nt_sG_converged = calc_ref.density.nt_sG.copy()
    
    # D_asp handling depends on GPAW version
    # Try to deep copy the density matrix
    try:
        D_asp_converged = {}
        for a, D_sp in calc_ref.density.D_asp.items():
            D_asp_converged[a] = D_sp.copy()
        has_D_asp = True
        print(f"  Saved D_asp for {len(D_asp_converged)} atoms")
    except Exception as e:
        print(f"  Could not copy D_asp: {e}")
        has_D_asp = False

    h = 0.2
    dv = h**3
    n_pseudo_ext = calc_ref.get_pseudo_density(pad=True)
    n_ae_ext = calc_ref.get_all_electron_density(gridrefinement=1, pad=True)

    print(f"  nt_sG shape: {nt_sG_converged.shape}")
    print(f"  nt_sG range: [{nt_sG_converged.min():.6f}, {nt_sG_converged.max():.6f}]")
    print(f"  Pseudo (padded) ∫ñdV = {n_pseudo_ext.sum() * dv:.4f}")
    print(f"  AE (padded) ∫n_AE dV = {n_ae_ext.sum() * dv:.4f}")

    # Clean up reference calculator
    atoms_ref.calc = None

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Fresh GPAW → inject density → single-shot
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- STEP 3: Fresh GPAW → inject converged density → 1 SCF step ---")

    atoms_test = atoms.copy()
    calc_test = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False,
                     maxiter=1)  # Force exactly 1 SCF iteration
    atoms_test.calc = calc_test

    # We need to initialize the calculator first so all PAW setups are ready
    # Then replace the density before the SCF loop runs
    # GPAW's initialize() sets up grids, PAW, etc. without running SCF
    calc_test.initialize(atoms_test)
    calc_test.set_positions(atoms_test)

    # Now inject the converged density
    print(f"  Injecting converged nt_sG...")
    print(f"  calc_test.density.nt_sG shape: {calc_test.density.nt_sG.shape}")
    print(f"  Converged nt_sG shape: {nt_sG_converged.shape}")

    if calc_test.density.nt_sG.shape == nt_sG_converged.shape:
        calc_test.density.nt_sG[:] = nt_sG_converged
        print(f"  ✓ nt_sG injected successfully")
    else:
        print(f"  ✗ Shape mismatch! Cannot inject.")
        return

    if has_D_asp:
        try:
            for a, D_sp in D_asp_converged.items():
                calc_test.density.D_asp[a][:] = D_sp
            print(f"  ✓ D_asp injected successfully")
        except Exception as e:
            print(f"  ⚠ D_asp injection failed: {e}")
            print(f"  Proceeding with only nt_sG injection...")

    # Update the effective potential from injected density
    calc_test.density.calculate_pseudo_charge()
    
    # Now run the calculation (maxiter=1 → exactly 1 SCF step)
    print(f"  Running 1 SCF iteration...")
    try:
        e_test = atoms_test.get_potential_energy()
        converged_1step = True
    except Exception as e:
        # maxiter=1 might raise ConvergenceError, but eigenvalues should still be available
        print(f"  SCF did not converge (expected with maxiter=1): {e}")
        converged_1step = False
        try:
            e_test = calc_test.get_potential_energy()
        except:
            e_test = float('nan')

    # Extract eigenvalues after 1 step
    try:
        evals_test = calc_test.get_eigenvalues(spin=0)
        homo_test = evals_test[n_occupied - 1]
        lumo_test = evals_test[n_occupied]
        print(f"  ✓ Eigenvalues extracted after 1 step")
    except Exception as e:
        print(f"  ✗ Could not get eigenvalues: {e}")
        evals_test = None

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Also test with DEFAULT initial guess (baseline)
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- STEP 4: Fresh GPAW → DEFAULT density → 1 SCF step (baseline) ---")

    atoms_baseline = atoms.copy()
    calc_baseline = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False,
                         maxiter=1)
    atoms_baseline.calc = calc_baseline

    try:
        e_baseline = atoms_baseline.get_potential_energy()
    except:
        e_baseline = float('nan')

    try:
        evals_baseline = calc_baseline.get_eigenvalues(spin=0)
        homo_baseline = evals_baseline[n_occupied - 1]
        lumo_baseline = evals_baseline[n_occupied]
    except Exception as e:
        print(f"  Could not get baseline eigenvalues: {e}")
        evals_baseline = None

    atoms_baseline.calc = None

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Comparison
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Method':<30} {'HOMO (eV)':<12} {'LUMO (eV)':<12} {'Gap (eV)':<12} {'E_total (eV)':<14}")
    print(f"  {'-'*80}")

    print(f"  {'Full SCF (ground truth)':<30} {homo_ref:<12.4f} {lumo_ref:<12.4f} {lumo_ref-homo_ref:<12.4f} {e_ref:<14.6f}")

    if evals_test is not None:
        gap_test = lumo_test - homo_test
        print(f"  {'1-step (injected density)':<30} {homo_test:<12.4f} {lumo_test:<12.4f} {gap_test:<12.4f} {e_test:<14.6f}")
        print(f"  {'Δ (injected - truth)':<30} {homo_test-homo_ref:<12.4f} {lumo_test-lumo_ref:<12.4f} {gap_test-(lumo_ref-homo_ref):<12.4f} {e_test-e_ref:<14.6f}")

    if evals_baseline is not None:
        gap_baseline = lumo_baseline - homo_baseline
        print(f"  {'1-step (default guess)':<30} {homo_baseline:<12.4f} {lumo_baseline:<12.4f} {gap_baseline:<12.4f} {e_baseline:<14.6f}")
        print(f"  {'Δ (default - truth)':<30} {homo_baseline-homo_ref:<12.4f} {lumo_baseline-lumo_ref:<12.4f} {gap_baseline-(lumo_ref-homo_ref):<12.4f} {e_baseline-e_ref:<14.6f}")

    if evals_test is not None and evals_baseline is not None:
        print(f"\n  Per-eigenvalue comparison (eV):")
        print(f"  {'Level':<8} {'Converged':<12} {'Injected':<12} {'Default':<12} {'Δ Inject':<12} {'Δ Default':<12}")
        print(f"  {'-'*68}")
        for i in range(min(len(evals_ref), len(evals_test), len(evals_baseline))):
            label = "HOMO" if i == n_occupied-1 else ("LUMO" if i == n_occupied else f"ε_{i+1}")
            print(f"  {label:<8} {evals_ref[i]:<12.4f} {evals_test[i]:<12.4f} {evals_baseline[i]:<12.4f} {evals_test[i]-evals_ref[i]:<12.4f} {evals_baseline[i]-evals_ref[i]:<12.4f}")

    print(f"\n{'='*70}")
    if evals_test is not None:
        max_error_injected = max(abs(evals_test[i] - evals_ref[i]) for i in range(min(len(evals_ref), len(evals_test))))
        print(f"  Max eigenvalue error (injected): {max_error_injected:.4f} eV")
        if max_error_injected < 0.5:
            print(f"  ✅ VERDICT: Pipeline WORKS! Single-shot from pseudo-density gives ~converged eigenvalues.")
            print(f"  → Retraining on pseudo-density is VIABLE for HOMO/LUMO prediction.")
        elif max_error_injected < 2.0:
            print(f"  ⚠️ VERDICT: Partial success. Eigenvalues within ~{max_error_injected:.1f} eV.")
            print(f"  → May need 2-3 SCF iterations instead of 1.")
        else:
            print(f"  ❌ VERDICT: Single-shot not sufficient. Errors too large.")
    print(f"{'='*70}")

    atoms_test.calc = None


if __name__ == '__main__':
    main()
