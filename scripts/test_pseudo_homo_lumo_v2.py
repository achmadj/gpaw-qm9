#!/usr/bin/env python3
"""
v2: Extract GPAW's internal smooth effective potential directly,
then diagonalize H = T + V_eff_smooth to get HOMO/LUMO.

Key insight from v1: The stored v_ext has -Z/r singularities (~-171 Hartree),
which is the BARE nuclear potential. GPAW internally uses a SMOOTH local
potential (regularized inside augmentation spheres). We need the latter.
"""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import h5py
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW

HARTREE_TO_EV = 27.211386245988


def build_laplacian_3d(shape, h):
    nx, ny, nz = shape
    D1x = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx))
    D1y = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(ny, ny))
    D1z = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz))
    L = sp.kronsum(sp.kronsum(D1x, D1y), D1z)
    return L


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
    print("GPAW INTERNAL POTENTIAL EXTRACTION + GRID DIAGONALIZATION")
    print("=" * 70)
    print(f"Molecule: {formula}, electrons: {num_electrons}, occupied: {n_occupied}")

    padding_angstrom = 4.0 * Bohr
    atoms.center(vacuum=padding_angstrom)

    # Run GPAW SCF
    calc = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False)
    atoms.calc = calc
    e_pot = atoms.get_potential_energy()

    # Ground truth eigenvalues
    homo, lumo = calc.get_homo_lumo()
    nbands = calc.get_number_of_bands()
    ks_evals_eV = calc.get_eigenvalues(spin=0)

    print(f"\n--- GPAW Converged ---")
    print(f"  HOMO: {homo:.4f} eV, LUMO: {lumo:.4f} eV, Gap: {lumo-homo:.4f} eV")
    print(f"  KS eigenvalues (eV): {ks_evals_eV}")

    # ─── Extract GPAW's internal smooth effective potential ───────────
    # In GPAW fd mode, the smooth effective potential is stored in:
    #   calc.hamiltonian.vt_sG  (spin, grid) in Hartree
    # This is the smooth local potential that GPAW actually uses for
    # diagonalization. It includes:
    #   ṽ_eff = ṽ_local + V_H[ñ+ñ_c] + V_xc[ñ]
    # where ṽ_local is the smooth local part of the PAW potential
    # (regularized inside augmentation spheres, NO -Z/r singularity)

    hamiltonian = calc.hamiltonian
    
    # Get the smooth effective potential on the coarse grid
    # vt_sG has shape (nspins, gd.n_c[0], gd.n_c[1], gd.n_c[2])
    # For non-spin-polarized: spin index 0
    vt_sG = hamiltonian.vt_sG
    print(f"\n--- GPAW Internal Potential ---")
    print(f"  vt_sG shape: {vt_sG.shape}")
    print(f"  vt_sG[0] range: [{vt_sG[0].min():.6f}, {vt_sG[0].max():.6f}] Hartree")
    print(f"  vt_sG[0] range: [{vt_sG[0].min()*HARTREE_TO_EV:.4f}, {vt_sG[0].max()*HARTREE_TO_EV:.4f}] eV")

    # This is the key: GPAW's smooth potential has NO singularities!
    # Compare with the bare v_ext which has -171 Hartree peaks
    v_smooth = np.array(vt_sG[0], dtype=np.float64)

    # Get pseudo-density for comparison
    n_pseudo = calc.get_pseudo_density(pad=True)
    n_ae = calc.get_all_electron_density(gridrefinement=1, pad=True)

    h = 0.2  # Bohr
    dv = h**3
    print(f"\n  Pseudo-density ∫ñdV = {n_pseudo.sum()*dv:.4f}")
    print(f"  All-electron  ∫n_AE dV = {n_ae.sum()*dv:.4f}")

    # ─── Check if vt_sG needs padding ────────────────────────────────
    # GPAW may not pad the potential the same way as get_pseudo_density(pad=True)
    # The internal grid gd might not include boundary padding
    gd = calc.wfs.gd  # grid descriptor
    print(f"\n  GPAW grid descriptor: {gd.N_c} (global), {gd.n_c} (local)")
    print(f"  Pseudo-density shape (padded): {n_pseudo.shape}")
    print(f"  Smooth potential shape: {v_smooth.shape}")

    # Use the smooth potential grid shape for diagonalization
    shape = v_smooth.shape
    N = np.prod(shape)
    n_states = n_occupied + 3  # HOMO + a few unoccupied

    print(f"\n--- Grid Diagonalization using GPAW's smooth V_eff ---")
    print(f"  Shape: {shape}, N={N}, solving for {n_states} states")

    L = build_laplacian_3d(shape, h)
    T = -0.5 * L / (h**2)
    V_diag = sp.diags(v_smooth.flatten())
    H = T + V_diag

    print(f"  Solving eigenvalue problem...")
    evals, evecs = spla.eigsh(H, k=n_states, which='SA', tol=1e-5)
    evals_eV = evals * HARTREE_TO_EV

    print(f"  Grid eigenvalues (Hartree): {evals}")
    print(f"  Grid eigenvalues (eV):      {evals_eV}")

    # ─── Comparison ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON: GPAW vs Grid Diagonalization")
    print(f"{'='*70}")
    print(f"\n  {'Level':<8} {'GPAW (eV)':<14} {'Grid (eV)':<14} {'Δ (eV)':<12}")
    print(f"  {'-'*48}")

    for i in range(min(n_states, len(ks_evals_eV))):
        if i == n_occupied - 1:
            label = "HOMO"
        elif i == n_occupied:
            label = "LUMO"
        else:
            label = f"ε_{i+1}"
        gpaw_val = ks_evals_eV[i]
        grid_val = evals_eV[i]
        delta = grid_val - gpaw_val
        print(f"  {label:<8} {gpaw_val:<14.4f} {grid_val:<14.4f} {delta:<12.4f}")

    homo_grid = evals_eV[n_occupied - 1]
    lumo_grid = evals_eV[n_occupied]
    gap_grid = lumo_grid - homo_grid
    gap_gpaw = lumo - homo

    print(f"\n  HOMO-LUMO gap:")
    print(f"    GPAW:  {gap_gpaw:.4f} eV")
    print(f"    Grid:  {gap_grid:.4f} eV")
    print(f"    Δ:     {gap_grid - gap_gpaw:.4f} eV")

    # ─── Key diagnostic: What the smooth potential looks like ────────
    print(f"\n--- Diagnostic: Potential Statistics ---")
    print(f"  GPAW smooth V_eff:")
    print(f"    min = {v_smooth.min():.4f} Ha = {v_smooth.min()*HARTREE_TO_EV:.2f} eV")
    print(f"    max = {v_smooth.max():.4f} Ha = {v_smooth.max()*HARTREE_TO_EV:.2f} eV")

    # Load stored v_ext for comparison
    merged_path = '/home/achmadjae/gpaw-qm9/dataset/gpaw_qm9_merged.h5'
    with h5py.File(merged_path, 'r') as f:
        v_ext_stored = f[mol_key]['v_ext'][:].astype(np.float64)
    print(f"  Stored bare v_ext:")
    print(f"    min = {v_ext_stored.min():.4f} Ha = {v_ext_stored.min()*HARTREE_TO_EV:.2f} eV")
    print(f"    max = {v_ext_stored.max():.4f} Ha = {v_ext_stored.max()*HARTREE_TO_EV:.2f} eV")

    print(f"\n  NOTE: GPAW's smooth potential has NO singularity!")
    print(f"  The PAW method replaces -Z/r inside augmentation spheres")
    print(f"  with a smooth function, making grid diagonalization feasible.")

    atoms.calc = None


if __name__ == '__main__':
    main()
