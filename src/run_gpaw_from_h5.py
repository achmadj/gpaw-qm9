import argparse
import os
import random
import sys
import time

import h5py
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw import GPAW


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for selection"
    )
    parser.add_argument(
        "--n_mols",
        type=int,
        default=3,
        help="Number of molecules to process; use 0 to process all",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="qm9_smallest_20k.h5",
        help="Path to QM9 HDF5 database",
    )
    parser.add_argument(
        "--out_path", type=str, default="gpaw_test_3_qm9.h5", help="Output HDF5 path"
    )
    parser.add_argument(
        "--selection",
        choices=["random", "all"],
        default="random",
        help="How to choose molecules from the source database",
    )
    parser.add_argument(
        "--padding_bohr",
        type=float,
        default=4.0,
        help="Vacuum padding around the outermost atom in Bohr",
    )
    parser.add_argument(
        "--h", type=float, default=0.2, help="GPAW real-space grid spacing in Angstrom"
    )
    parser.add_argument(
        "--xc", type=str, default="LDA", help="Exchange-correlation functional"
    )
    parser.add_argument("--mode", type=str, default="fd", help="GPAW mode")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip completed molecules",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Optional directory for per-molecule GPAW text logs",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="0-based shard index for parallel workers",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1, help="Total number of parallel shards"
    )
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

    with h5py.File(args.db_path, "r") as db:
        all_mol_keys = list(db.keys())
        selected_keys = select_molecule_keys(
            all_mol_keys, args.selection, args.n_mols, args.seed
        )
        shard_selected_keys = shard_keys(
            selected_keys, args.shard_index, args.num_shards
        )

        log(f"{worker_tag} Selection mode: {args.selection}")
        if args.selection == "random" and args.n_mols > 0:
            log(
                f"{worker_tag} Selected {len(selected_keys)} random molecules using seed {args.seed}: {selected_keys}"
            )
        else:
            log(f"{worker_tag} Selected all molecules: {len(selected_keys)}")
            log(f"{worker_tag} First 5 molecule keys: {selected_keys[:5]}")
        log(
            f"{worker_tag} Assigned molecules in this shard: {len(shard_selected_keys)}"
        )

        output_mode = "a" if args.resume and os.path.exists(args.out_path) else "w"
        with h5py.File(args.out_path, output_mode) as out_db:
            out_db.attrs["description"] = (
                "GPAW results generated from qm9_smallest_20k.h5"
            )
            out_db.attrs["seed"] = args.seed
            out_db.attrs["selection"] = args.selection
            out_db.attrs["padding_bohr"] = args.padding_bohr
            out_db.attrs["grid_spacing_angstrom"] = args.h
            out_db.attrs["xc"] = args.xc
            out_db.attrs["mode"] = args.mode
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
                    log(
                        f"{worker_tag} SKIP local={local_order}/{len(shard_selected_keys)} "
                        f"global={global_order}/{len(selected_keys)} key={mol_key}"
                    )
                    continue

                start_time = time.time()
                grp = db[mol_key]
                positions = grp["positions"][:]
                numbers = grp["numbers"][:]
                smiles = grp.attrs.get("smiles", "Unknown")

                atoms = Atoms(numbers=numbers, positions=positions)
                formula = atoms.get_chemical_formula()
                qm9_index = parse_qm9_index(mol_key)

                log(
                    f"{worker_tag} START local={local_order}/{len(shard_selected_keys)} "
                    f"global={global_order}/{len(selected_keys)} index={qm9_index} name={formula} atoms={len(atoms)}"
                )

                padding_angstrom = args.padding_bohr * Bohr
                atoms.center(vacuum=padding_angstrom)
                log(
                    f"{worker_tag} INFO index={qm9_index} padding={args.padding_bohr}bohr h={args.h}"
                )

                log_txt = (
                    os.path.join(args.log_dir, f"log_gpaw_{mol_key}.txt")
                    if args.log_dir
                    else None
                )
                calc = GPAW(
                    h=args.h, mode=args.mode, xc=args.xc, txt=log_txt, spinpol=False
                )
                atoms.calc = calc

                try:
                    e_pot = atoms.get_potential_energy()
                    e_ref = calc.get_reference_energy()
                    e_total = e_pot + e_ref

                    # Export all-electron density on the coarse/padded GPAW grid.
                    # gridrefinement=1 + pad=True gives a shape matching the pseudo-density grid
                    # while integrating to the total number of electrons.
                    n_r = calc.get_all_electron_density(gridrefinement=1, pad=True)

                    # GPAW electrostatic potential is returned on the refined grid.
                    # Downsample by 2x2x2 mean so that v_ext matches n_r.shape.
                    v_ext_full = calc.get_electrostatic_potential()
                    if v_ext_full.shape != n_r.shape:
                        expected = tuple(2 * s for s in n_r.shape)
                        if tuple(v_ext_full.shape) != expected:
                            raise ValueError(
                                f"Unexpected v_ext shape {v_ext_full.shape}; "
                                f"expected either {n_r.shape} or {expected} for n_r shape {n_r.shape}"
                            )
                        v_ext = downsample_mean_2x2x2(v_ext_full)
                    else:
                        v_ext = v_ext_full

                    elapsed = time.time() - start_time
                    
                    # Convert to float16 to massively save disk space while preserving exponential range
                    n_r_fp16 = n_r.astype(np.float16)
                    v_ext_fp16 = v_ext.astype(np.float16)

                    out_grp = out_db.create_group(mol_key)
                    out_grp.create_dataset(
                        "n_r",
                        data=n_r_fp16,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
                    out_grp.create_dataset(
                        "v_ext",
                        data=v_ext_fp16,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

                    out_grp.attrs["index"] = qm9_index
                    out_grp.attrs["selection_order"] = global_order - 1
                    out_grp.attrs["elapsed_time"] = elapsed
                    out_grp.attrs["total_energy_hartree"] = e_total / Hartree
                    out_grp.attrs["smiles"] = smiles
                    out_grp.attrs["formula"] = formula
                    out_grp.attrs["num_atoms"] = len(atoms)
                    out_grp.attrs["padding_bohr"] = args.padding_bohr
                    out_grp.attrs["grid_spacing_angstrom"] = args.h
                    out_grp.attrs["cell_angstrom"] = atoms.cell.array
                    out_grp.attrs["density_kind"] = "all_electron"
                    out_grp.attrs["density_gridrefinement"] = 1
                    out_grp.attrs["density_pad"] = True
                    out_grp.attrs["v_ext_grid_match"] = "downsample_to_n_r_if_needed"
                    out_grp.attrs["worker_id"] = args.shard_index

                    out_db.flush()
                    processed += 1
                    log(
                        f"{worker_tag} DONE index={qm9_index} name={formula} elapsed={elapsed:.1f}s "
                        f"energy={e_total / Hartree:.6f}Ha n_r={n_r.shape} v_ext={v_ext.shape}"
                    )

                except Exception as exc:
                    log(
                        f"{worker_tag} FAIL index={qm9_index} name={formula} key={mol_key} error={exc}"
                    )

                finally:
                    atoms.calc = None

            log(
                f"{worker_tag} FINISHED processed={processed} skipped={skipped} "
                f"assigned={len(shard_selected_keys)} total_selected={len(selected_keys)}"
            )


if __name__ == "__main__":
    main()
