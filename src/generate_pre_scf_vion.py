"""Generate pre-SCF ionic potential on grid using GPAW maxiter=0."""
import argparse, os, time
import h5py, numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--padding_bohr", type=float, default=4.0)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--xc", default="LDA")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    with h5py.File(args.db_path, "r") as db:
        all_keys = sorted(db.keys())
        shard_keys = all_keys[args.shard_index::args.num_shards]

        os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
        with h5py.File(args.out_path, "w") as out:
            for i, key in enumerate(shard_keys):
                grp = db[key]
                positions = grp["positions"][:]
                numbers = grp["numbers"][:]
                atoms = Atoms(numbers=numbers, positions=positions)
                atoms.center(vacuum=args.padding_bohr * Bohr)

                calc = GPAW(h=args.h, mode="fd", xc=args.xc,
                            maxiter=0, txt=None, spinpol=False)
                atoms.calc = calc

                try:
                    atoms.get_potential_energy()
                except Exception:
                    pass  # Expected: SCF did not converge (0 iterations)

                vt = calc.get_effective_potential()

                out_grp = out.create_group(key)
                out_grp.create_dataset("v_ion", data=vt.astype(np.float16),
                                       compression="gzip", compression_opts=4)
                out_grp.attrs["formula"] = atoms.get_chemical_formula()
                out_grp.attrs["num_atoms"] = len(atoms)
                out_grp.attrs["cell_angstrom"] = atoms.cell.array
                out_grp.attrs["grid_spacing_angstrom"] = args.h

                atoms.calc = None
                print(f"[{i+1}/{len(shard_keys)}] {key} shape={vt.shape}")

if __name__ == "__main__":
    main()
