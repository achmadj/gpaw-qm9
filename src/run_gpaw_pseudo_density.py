"""Generate pseudo-density dataset from QM9 molecules using GPAW.

This is the pseudo-density counterpart of run_gpaw_from_h5.py.
Instead of saving the all-electron density (with nuclear cusps),
it saves the smooth pseudo-density ñ(r) which is compatible with
single-shot KS diagonalization for HOMO-LUMO extraction.

Output HDF5 structure (per molecule):
  ├── n_pseudo      (float16, 3D)  — smooth PAW pseudo-density
  ├── v_ext         (float16, 3D)  — bare external potential (same as AE dataset)
  └── attrs:
      ├── homo_eV, lumo_eV, gap_eV     — converged KS eigenvalues
      ├── eigenvalues_eV                — all KS eigenvalues
      ├── total_energy_hartree          — total energy
      ├── n_pseudo_integral             — ∫ñ(r)dV (should ≈ valence electrons)
      ├── n_electrons_total             — total electron count
      ├── formula, smiles, cell, etc.   — metadata
      └── density_kind = "pseudo"

Usage:
    python src/run_gpaw_pseudo_density.py \\
        --db_path dataset/raw/qm9_smallest_20k.h5 \\
        --out_path dataset/shards/gpaw_pseudo_shard_0.h5 \\
        --n_mols 0 --selection all \\
        --shard_index 0 --num_shards 6 \\
        --padding_bohr 4.0 --h 0.2 --resume
"""

import argparse
import os

# --- PREVENT SLURM MPI CONTEXT LEAKAGE ---
# When running with 'srun --ntasks=X', SLURM creates an MPI context.
# mpi4py automatically detects this and groups all independent shard workers 
# into a single MPI world, causing 'Mismatch of Atoms' errors. 
# We purge MPI/PMIX variables so each Python process runs in pure serial mode.
for _k in list(os.environ.keys()):
    if _k.startswith(("PMIX_", "PMI_", "SLURM_MPI_", "SLURM_PMIX")):
        os.environ.pop(_k, None)

import random
import sys
import time

import h5py
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW


ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
VALENCE_ELECTRONS = {"H": 1, "C": 4, "N": 5, "O": 6, "F": 7}
VALENCE_BY_Z = {ATOMIC_NUMBERS[sym]: val for sym, val in VALENCE_ELECTRONS.items()}


def parse_qm9_index(mol_key):
    try:
        return int(mol_key.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def select_molecule_keys(all_keys, selection, n_mols, seed):
    ordered_keys = sorted(all_keys)
    if selection == "all" or n_mols <= 0 or n_mols >= len(ordered_keys):
        return ordered_keys
    rng = random.Random(seed)
    return rng.sample(ordered_keys, n_mols)


def shard_keys(selected_keys, shard_index, num_shards):
    if num_shards <= 1:
        return list(selected_keys)
    return list(selected_keys[shard_index::num_shards])


def log(message):
    print(message, flush=True)


def downsample_mean_2x2x2(arr):
    """Downsample a 3D array by averaging 2×2×2 blocks."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    nx, ny, nz = arr.shape
    if nx % 2 != 0 or ny % 2 != 0 or nz % 2 != 0:
        raise ValueError(
            f"Array shape must be even along all axes for 2x2x2 downsampling: {arr.shape}"
        )
    return arr.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2).mean(
        axis=(1, 3, 5), dtype=np.float64
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Generate pseudo-density dataset from QM9 using GPAW"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mols", type=int, default=3,
                        help="0 = process all molecules")
    parser.add_argument("--db_path", type=str,
                        default="dataset/raw/qm9_smallest_20k.h5")
    parser.add_argument("--out_path", type=str,
                        default="dataset/shards/gpaw_pseudo_shard_0.h5")
    parser.add_argument("--selection", choices=["random", "all"], default="random")
    parser.add_argument("--padding_bohr", type=float, default=4.0)
    parser.add_argument("--h", type=float, default=0.2,
                        help="Grid spacing in Angstrom")
    parser.add_argument("--xc", type=str, default="LDA")
    parser.add_argument("--mode", type=str, default="fd")
    parser.add_argument(
        "--setups",
        type=str,
        default="",
        help="Optional GPAW setup mode (e.g. 'hgh', 'sg15'). Empty keeps GPAW default.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must be within [0, num_shards)")

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    worker_tag = f"[worker {args.shard_index + 1}/{args.num_shards}]"
    log(f"{worker_tag} Opening database: {args.db_path}")
    setups_label = args.setups.strip() if args.setups else ""
    log(f"{worker_tag} GPAW config: mode={args.mode} xc={args.xc} "
        f"setups={(setups_label if setups_label else 'default')}")

    with h5py.File(args.db_path, "r") as db:
        all_mol_keys = list(db.keys())
        selected_keys = select_molecule_keys(
            all_mol_keys, args.selection, args.n_mols, args.seed
        )
        shard_selected_keys = shard_keys(
            selected_keys, args.shard_index, args.num_shards
        )

        log(f"{worker_tag} Selection: {args.selection}, "
            f"total={len(selected_keys)}, shard={len(shard_selected_keys)}")

        output_mode = "a" if args.resume and os.path.exists(args.out_path) else "w"
        with h5py.File(args.out_path, output_mode) as out_db:
            out_db.attrs["description"] = (
                "GPAW pseudo-density dataset from QM9 for single-shot KS evaluation"
            )
            out_db.attrs["seed"] = args.seed
            out_db.attrs["selection"] = args.selection
            out_db.attrs["padding_bohr"] = args.padding_bohr
            out_db.attrs["grid_spacing_angstrom"] = args.h
            out_db.attrs["xc"] = args.xc
            out_db.attrs["mode"] = args.mode
            out_db.attrs["setups"] = setups_label if setups_label else "default"
            out_db.attrs["source_db"] = args.db_path
            out_db.attrs["shard_index"] = args.shard_index
            out_db.attrs["num_shards"] = args.num_shards

            processed = 0
            skipped = 0

            for local_order, mol_key in enumerate(shard_selected_keys, start=1):
                global_order = (
                    args.shard_index + (local_order - 1) * args.num_shards + 1
                )

                if mol_key in out_db:
                    skipped += 1
                    log(f"{worker_tag} SKIP {local_order}/{len(shard_selected_keys)} "
                        f"key={mol_key}")
                    continue

                start_time = time.time()
                grp = db[mol_key]
                positions = grp["positions"][:]
                numbers = grp["numbers"][:]
                smiles = grp.attrs.get("smiles", "Unknown")

                atoms = Atoms(numbers=numbers, positions=positions)
                formula = atoms.get_chemical_formula()
                qm9_index = parse_qm9_index(mol_key)
                n_electrons_total = int(sum(numbers))
                n_electrons_valence = int(sum(VALENCE_BY_Z[int(z)] for z in numbers))
                n_occupied = n_electrons_valence // 2

                log(f"{worker_tag} START {local_order}/{len(shard_selected_keys)} "
                    f"idx={qm9_index} {formula} atoms={len(atoms)}")

                padding_angstrom = args.padding_bohr * Bohr
                atoms.center(vacuum=padding_angstrom)

                log_txt = (
                    os.path.join(args.log_dir, f"log_gpaw_pseudo_{mol_key}.txt")
                    if args.log_dir
                    else None
                )
                calc_kwargs = {
                    "h": args.h,
                    "mode": args.mode,
                    "xc": args.xc,
                    "txt": log_txt,
                    "spinpol": False,
                }
                if setups_label:
                    calc_kwargs["setups"] = setups_label
                calc = GPAW(**calc_kwargs)
                atoms.calc = calc

                try:
                    e_pot = atoms.get_potential_energy()
                    e_ref = calc.get_reference_energy()
                    e_total = e_pot + e_ref

                    # ── Pseudo-density (smooth, no cusps) ─────────────
                    n_pseudo = calc.get_pseudo_density(pad=True)

                    # ── PAW on-site density matrices (D_asp) ──────────
                    D_asp_data = {}
                    for a, D_sp in calc.density.D_asp.items():
                        D_asp_data[str(a)] = np.array(D_sp, dtype=np.float64)

                    # ── External potential (same as AE dataset) ───────
                    v_ext_full = calc.get_electrostatic_potential()
                    if v_ext_full.shape != n_pseudo.shape:
                        expected = tuple(2 * s for s in n_pseudo.shape)
                        if tuple(v_ext_full.shape) != expected:
                            raise ValueError(
                                f"v_ext shape {v_ext_full.shape} unexpected; "
                                f"expected {n_pseudo.shape} or {expected}"
                            )
                        v_ext = downsample_mean_2x2x2(v_ext_full)
                    else:
                        v_ext = v_ext_full

                    # ── KS eigenvalues (HOMO/LUMO/gap) ────────────────
                    evals = calc.get_eigenvalues(spin=0)
                    homo = evals[n_occupied - 1]
                    if n_occupied < len(evals):
                        lumo = evals[n_occupied]
                        gap = lumo - homo
                    else:
                        lumo = float('nan')
                        gap = float('nan')

                    # ── Integral of pseudo-density ────────────────────
                    # get_pseudo_density(pad=True) returns array in units of e/Angstrom^3
                    # But gd.dv is in Bohr^3. To get the number of electrons,
                    # we must convert gd.dv to Angstrom^3 by multiplying by Bohr^3
                    h_bohr = calc.wfs.gd.h_cv.diagonal()  # grid spacings in Bohr
                    dv_bohr = float(np.prod(h_bohr))
                    dv_angstrom = dv_bohr * (Bohr ** 3)
                    n_pseudo_integral = float(n_pseudo.sum() * dv_angstrom)

                    elapsed = time.time() - start_time

                    # ── Save to HDF5 ──────────────────────────────────
                    n_pseudo_fp16 = n_pseudo.astype(np.float16)
                    v_ext_fp16 = v_ext.astype(np.float16)

                    out_grp = out_db.create_group(mol_key)
                    out_grp.create_dataset(
                        "n_pseudo", data=n_pseudo_fp16,
                        compression="gzip", compression_opts=4, shuffle=True,
                    )
                    out_grp.create_dataset(
                        "v_ext", data=v_ext_fp16,
                        compression="gzip", compression_opts=4, shuffle=True,
                    )

                    # Save D_asp subgroup
                    dasp_grp = out_grp.create_group("D_asp")
                    for atom_idx_str, d_matrix in D_asp_data.items():
                        dasp_grp.create_dataset(
                            atom_idx_str, data=d_matrix,
                            compression="gzip", compression_opts=4,
                        )

                    out_grp.attrs["index"] = qm9_index
                    out_grp.attrs["selection_order"] = global_order - 1
                    out_grp.attrs["elapsed_time"] = elapsed
                    out_grp.attrs["total_energy_hartree"] = e_total / Hartree
                    out_grp.attrs["smiles"] = smiles
                    out_grp.attrs["formula"] = formula
                    out_grp.attrs["num_atoms"] = len(atoms)
                    out_grp.attrs["n_electrons_total"] = n_electrons_total
                    out_grp.attrs["n_electrons_valence"] = n_electrons_valence
                    out_grp.attrs["n_occupied"] = n_occupied
                    out_grp.attrs["padding_bohr"] = args.padding_bohr
                    out_grp.attrs["grid_spacing_angstrom"] = args.h
                    out_grp.attrs["xc"] = args.xc
                    out_grp.attrs["mode"] = args.mode
                    out_grp.attrs["setups"] = setups_label if setups_label else "default"
                    out_grp.attrs["cell_angstrom"] = atoms.cell.array
                    out_grp.attrs["density_kind"] = "pseudo"
                    out_grp.attrs["density_pad"] = True
                    out_grp.attrs["n_pseudo_integral"] = n_pseudo_integral
                    out_grp.attrs["homo_eV"] = homo
                    out_grp.attrs["lumo_eV"] = lumo
                    out_grp.attrs["gap_eV"] = gap
                    out_grp.attrs["eigenvalues_eV"] = evals
                    out_grp.attrs["v_ext_grid_match"] = "downsample_to_n_pseudo_if_needed"
                    out_grp.attrs["worker_id"] = args.shard_index

                    out_db.flush()
                    processed += 1
                    log(
                        f"{worker_tag} DONE idx={qm9_index} {formula} "
                        f"{elapsed:.1f}s E={e_total/Hartree:.6f}Ha "
                        f"HOMO={homo:.3f} LUMO={lumo:.3f} gap={gap:.3f}eV "
                        f"∫ñ={n_pseudo_integral:.2f} shape={n_pseudo.shape}"
                    )

                except Exception as exc:
                    log(f"{worker_tag} FAIL idx={qm9_index} {formula} "
                        f"key={mol_key} error={exc}")

                finally:
                    atoms.calc = None

            log(f"{worker_tag} FINISHED processed={processed} skipped={skipped} "
                f"assigned={len(shard_selected_keys)}")


if __name__ == "__main__":
    main()
