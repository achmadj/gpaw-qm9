#!/usr/bin/env python3
"""
Verify that pseudo-density → V_eff → diagonalize → HOMO/LUMO works.

This script:
1. Loads CH4 from the QM9 source HDF5
2. Runs GPAW SCF → gets converged eigenvalues (HOMO, LUMO) as ground truth
3. Extracts pseudo-density ñ(r) and all-electron density n_AE(r)
4. Constructs V_eff from pseudo-density via grid-based Poisson + LDA
5. Diagonalizes H = T + V_eff using scipy sparse eigensolver
6. Compares eigenvalues with GPAW's converged values
7. Repeats for all-electron density to show it fails

This is the key experiment: pseudo-density should give reasonable eigenvalues,
all-electron density should give garbage (proving the analysis correct).
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import h5py
from ase import Atoms
from ase.units import Bohr, Hartree

# Try importing GPAW
try:
    from gpaw import GPAW
    HAS_GPAW = True
except ImportError:
    HAS_GPAW = False
    print("WARNING: GPAW not available, will use stored data only")

HARTREE_TO_EV = 27.211386245988

# ─── Grid-based operators ────────────────────────────────────────────

def solve_poisson_fft(n_r, h):
    """Solve ∇²V_H = -4π n using FFT with zero-padding."""
    nx, ny, nz = n_r.shape
    pad_n = np.pad(n_r, ((nx//2, nx//2), (ny//2, ny//2), (nz//2, nz//2)),
                   mode='constant')
    nxf, nyf, nzf = pad_n.shape

    kx = np.fft.fftfreq(nxf, d=h) * 2 * np.pi
    ky = np.fft.fftfreq(nyf, d=h) * 2 * np.pi
    kz = np.fft.fftfreq(nzf, d=h) * 2 * np.pi

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0, 0, 0] = 1.0

    rho_k = np.fft.fftn(pad_n)
    V_k = 4 * np.pi * rho_k / K2
    V_k[0, 0, 0] = 0.0

    V_r = np.real(np.fft.ifftn(V_k))
    return V_r[nx//2:nx//2+nx, ny//2:ny//2+ny, nz//2:nz//2+nz]


def lda_vxc(n_r):
    """LDA exchange potential V_x = -(3/π)^(1/3) n^(1/3). No correlation."""
    n_safe = np.clip(n_r, 0, None)
    return -(3.0 / np.pi)**(1.0/3.0) * n_safe**(1.0/3.0)


def build_laplacian_3d(shape, h):
    """Build 3D Laplacian as sparse matrix using Kronecker sums."""
    nx, ny, nz = shape
    D1x = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx))
    D1y = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny))
    D1z = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz))
    L = sp.kronsum(sp.kronsum(D1x, D1y), D1z)
    return L


def diagonalize_from_density(density, v_ext, h, n_states, label=""):
    """
    Given a density field, construct V_eff and diagonalize H = T + V_eff.
    Returns eigenvalues in Hartree.
    """
    density = density.astype(np.float64)
    v_ext = v_ext.astype(np.float64)

    shape = density.shape
    N = np.prod(shape)

    print(f"\n{'='*60}")
    print(f"Diagonalizing: {label}")
    print(f"{'='*60}")
    print(f"  Grid shape: {shape}, total voxels: {N}")
    print(f"  Density range: [{density.min():.4f}, {density.max():.4f}] e/Bohr³")
    print(f"  V_ext range: [{v_ext.min():.4f}, {v_ext.max():.4f}] Hartree")

    dv = h**3
    n_e_integral = density.sum() * dv
    print(f"  ∫n(r)dV = {n_e_integral:.4f} electrons")

    # Construct V_eff = V_ext + V_H[n] + V_xc[n]
    V_H = solve_poisson_fft(density, h)
    V_xc = lda_vxc(density)
    V_eff = v_ext + V_H + V_xc

    print(f"  V_H range: [{V_H.min():.4f}, {V_H.max():.4f}]")
    print(f"  V_xc range: [{V_xc.min():.4f}, {V_xc.max():.4f}]")
    print(f"  V_eff range: [{V_eff.min():.4f}, {V_eff.max():.4f}]")

    # Build H = T + V_eff
    print(f"  Building Hamiltonian ({N}×{N} sparse)...")
    L = build_laplacian_3d(shape, h)
    T = -0.5 * L / (h**2)
    V_diag = sp.diags(V_eff.flatten())
    H = T + V_diag

    # Diagonalize
    print(f"  Solving for {n_states} lowest eigenvalues...")
    try:
        evals, evecs = spla.eigsh(H, k=n_states, which='SA', tol=1e-4)
        print(f"  Eigenvalues (Hartree): {evals}")
        print(f"  Eigenvalues (eV):      {evals * HARTREE_TO_EV}")
        return evals
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def main():
    # ─── Step 1: Run GPAW from QM9 source ────────────────────────────
    db_path = '/home/achmadjae/gpaw-qm9/dataset/raw/qm9_smallest_20k.h5'
    mol_key = 'dsgdb9nsd_000001'  # CH4

    print("=" * 60)
    print("PSEUDO-DENSITY vs ALL-ELECTRON: HOMO-LUMO TEST")
    print("=" * 60)

    # Load molecule
    with h5py.File(db_path, 'r') as db:
        grp = db[mol_key]
        positions = grp['positions'][:]
        numbers = grp['numbers'][:]

    atoms = Atoms(numbers=numbers, positions=positions)
    formula = atoms.get_chemical_formula()
    num_electrons = sum(numbers)
    print(f"Molecule: {formula} ({mol_key})")
    print(f"Total electrons: {num_electrons}")

    padding_angstrom = 4.0 * Bohr
    atoms.center(vacuum=padding_angstrom)

    if not HAS_GPAW:
        print("GPAW not available. Cannot proceed.")
        sys.exit(1)

    # Run GPAW SCF
    print("\n--- Running GPAW SCF ---")
    calc = GPAW(h=0.2, mode='fd', xc='LDA', txt=None, spinpol=False)
    atoms.calc = calc
    e_pot = atoms.get_potential_energy()
    e_ref = calc.get_reference_energy()
    e_total = (e_pot + e_ref) / Hartree

    # Get HOMO/LUMO from GPAW (ground truth)
    homo, lumo = calc.get_homo_lumo()
    print(f"\n--- GPAW Converged Results ---")
    print(f"  Total energy: {e_total:.6f} Hartree = {e_total * HARTREE_TO_EV:.4f} eV")
    print(f"  HOMO: {homo:.4f} eV")
    print(f"  LUMO: {lumo:.4f} eV")
    print(f"  HOMO-LUMO gap: {lumo - homo:.4f} eV")

    # Get all KS eigenvalues
    nbands = calc.get_number_of_bands()
    ks_eigenvalues_eV = np.array([calc.get_eigenvalues(spin=0)[i] for i in range(nbands)])
    print(f"  All KS eigenvalues (eV): {ks_eigenvalues_eV}")

    # ─── Step 2: Extract densities ────────────────────────────────────
    n_pseudo = calc.get_pseudo_density(pad=True)
    n_ae = calc.get_all_electron_density(gridrefinement=1, pad=True)
    v_ext_refined = calc.get_electrostatic_potential()

    # v_ext might be on refined grid, downsample if needed
    if v_ext_refined.shape != n_pseudo.shape:
        # Downsample 2x2x2
        nx, ny, nz = n_pseudo.shape
        v_ext = v_ext_refined.reshape(nx, 2, ny, 2, nz, 2).mean(axis=(1, 3, 5))
    else:
        v_ext = v_ext_refined

    # Convert v_ext from eV to Hartree (GPAW returns eV)
    v_ext_hartree = v_ext / HARTREE_TO_EV

    h = 0.2  # Bohr
    dv = h**3

    print(f"\n--- Density Comparison ---")
    print(f"  Pseudo-density shape: {n_pseudo.shape}")
    print(f"  Pseudo-density range: [{n_pseudo.min():.4f}, {n_pseudo.max():.4f}]")
    print(f"  Pseudo-density ∫ñdV = {n_pseudo.sum() * dv:.4f}")
    print(f"  All-electron shape: {n_ae.shape}")
    print(f"  All-electron range: [{n_ae.min():.4f}, {n_ae.max():.4f}]")
    print(f"  All-electron ∫n_AE dV = {n_ae.sum() * dv:.4f}")
    print(f"  Expected total electrons: {num_electrons}")

    # Check if grid is small enough for diagonalization
    N = np.prod(n_pseudo.shape)
    print(f"\n  Total grid points: {N}")
    if N > 200000:
        print(f"  WARNING: {N} grid points is very large for sparse eigsh.")
        print(f"  This may take a long time or run out of memory.")
        print(f"  Consider reducing vacuum padding or increasing h.")

    # ─── Step 3: Determine number of states needed ────────────────────
    # For CH4: 10 electrons, 5 occupied orbitals (spin-paired)
    n_occupied = num_electrons // 2
    n_states = n_occupied + 2  # HOMO, LUMO, LUMO+1

    # ─── Step 4: Grid-based diagonalization from PSEUDO density ───────
    # NOTE: We need the bare nuclear potential, not the full electrostatic potential
    # The electrostatic potential from GPAW includes V_H already.
    # We need V_ext (bare nuclear) = electrostatic_potential - V_H[n_pseudo]
    # Actually, let's use the stored v_ext from the merged dataset instead.

    # Load v_ext from the merged dataset (this is the bare nuclear potential)
    merged_path = '/home/achmadjae/gpaw-qm9/dataset/gpaw_qm9_merged.h5'
    if os.path.exists(merged_path):
        with h5py.File(merged_path, 'r') as f:
            if mol_key in f:
                v_ext_stored = f[mol_key]['v_ext'][:].astype(np.float64)
                n_ae_stored = f[mol_key]['n_r'][:].astype(np.float64)
                print(f"\n  Loaded stored v_ext shape: {v_ext_stored.shape}")
                print(f"  Loaded stored n_r (AE) shape: {n_ae_stored.shape}")

                # Check if shapes match
                if v_ext_stored.shape != n_pseudo.shape:
                    print(f"  Shape mismatch: stored {v_ext_stored.shape} vs fresh {n_pseudo.shape}")
                    print(f"  Will use GPAW electrostatic potential instead")
                    # Use bare nuclear potential from v_ext
                    # Actually calc.get_electrostatic_potential() returns the TOTAL
                    # electrostatic potential (V_ext + V_H). We need just V_ext.
                    # Let's compute it: V_ext = V_total - V_H
                    V_H_pseudo = solve_poisson_fft(n_pseudo, h)
                    v_ext_bare = v_ext_hartree - V_H_pseudo
                    use_stored = False
                else:
                    v_ext_bare = v_ext_stored
                    use_stored = True
            else:
                print(f"  {mol_key} not found in merged dataset")
                V_H_pseudo = solve_poisson_fft(n_pseudo, h)
                v_ext_bare = v_ext_hartree - V_H_pseudo
                use_stored = False
    else:
        print(f"  Merged dataset not found, computing V_ext from GPAW potential")
        V_H_pseudo = solve_poisson_fft(n_pseudo, h)
        v_ext_bare = v_ext_hartree - V_H_pseudo
        use_stored = False

    print(f"\n  V_ext (bare nuclear) range: [{v_ext_bare.min():.4f}, {v_ext_bare.max():.4f}] Hartree")

    # Diagonalize with pseudo-density
    evals_pseudo = diagonalize_from_density(
        n_pseudo, v_ext_bare, h, n_states,
        label="PSEUDO-DENSITY (smooth, should work)"
    )

    # Diagonalize with all-electron density
    evals_ae = diagonalize_from_density(
        n_ae, v_ext_bare, h, n_states,
        label="ALL-ELECTRON DENSITY (cusps, expected to fail)"
    )

    # ─── Step 5: Compare results ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    ks_hartree = ks_eigenvalues_eV / HARTREE_TO_EV

    print(f"\n  {'Level':<10} {'GPAW (eV)':<15} {'Pseudo (eV)':<15} {'AE (eV)':<15} {'Δ Pseudo':<12} {'Δ AE':<12}")
    print(f"  {'-'*74}")

    labels = []
    for i in range(min(n_states, len(ks_eigenvalues_eV))):
        if i == n_occupied - 1:
            label = f"HOMO"
        elif i == n_occupied:
            label = f"LUMO"
        else:
            label = f"ε_{i+1}"
        labels.append(label)

        gpaw_eV = ks_eigenvalues_eV[i]
        pseudo_eV = evals_pseudo[i] * HARTREE_TO_EV if evals_pseudo is not None else float('nan')
        ae_eV = evals_ae[i] * HARTREE_TO_EV if evals_ae is not None else float('nan')
        delta_pseudo = pseudo_eV - gpaw_eV
        delta_ae = ae_eV - gpaw_eV

        print(f"  {label:<10} {gpaw_eV:<15.4f} {pseudo_eV:<15.4f} {ae_eV:<15.4f} {delta_pseudo:<12.4f} {delta_ae:<12.4f}")

    if evals_pseudo is not None and len(evals_pseudo) > n_occupied:
        homo_pseudo = evals_pseudo[n_occupied - 1] * HARTREE_TO_EV
        lumo_pseudo = evals_pseudo[n_occupied] * HARTREE_TO_EV
        gap_pseudo = lumo_pseudo - homo_pseudo
        print(f"\n  Grid-diag HOMO-LUMO gap (pseudo): {gap_pseudo:.4f} eV")
        print(f"  GPAW HOMO-LUMO gap:               {lumo - homo:.4f} eV")
        print(f"  Difference:                        {gap_pseudo - (lumo - homo):.4f} eV")

    if evals_ae is not None and len(evals_ae) > n_occupied:
        homo_ae = evals_ae[n_occupied - 1] * HARTREE_TO_EV
        lumo_ae = evals_ae[n_occupied] * HARTREE_TO_EV
        gap_ae = lumo_ae - homo_ae
        print(f"\n  Grid-diag HOMO-LUMO gap (AE):     {gap_ae:.4f} eV")
        print(f"  GPAW HOMO-LUMO gap:               {lumo - homo:.4f} eV")
        print(f"  Difference:                        {gap_ae - (lumo - homo):.4f} eV")

    # Clean up
    atoms.calc = None


if __name__ == '__main__':
    main()
