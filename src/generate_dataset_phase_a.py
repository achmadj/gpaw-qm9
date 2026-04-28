"""
Phase A dataset generation: v_ion (pre-SCF, maxiter=0) + n_pseudo (full SCF)
from the 1000 smallest QM9 molecules.

Usage:
  conda activate gpaw
  python src/generate_dataset_phase_a.py \
      --sdf data/raw/gdb9.sdf \
      --out dataset/qm9_1000_phase_a.h5 \
      --n_mols 1000 --h 0.2 --padding_bohr 4.0
"""
import argparse, os, sys, time, re
import h5py
import numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW


# ── QM9 SDF parser ──────────────────────────────────────────────────
ELEMENT_MAP = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def parse_sdf_molecules(sdf_path, max_mols=None):
    """Parse an SDF file and return list of (name, numbers, positions)."""
    molecules = []
    with open(sdf_path, "r") as f:
        content = f.read()

    blocks = content.split("$$$$")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if len(lines) < 4:
            continue

        name = lines[0].strip()
        # counts line: "  5  4  0 ..." -> num_atoms, num_bonds
        counts = lines[3].strip().split()
        try:
            num_atoms = int(counts[0])
        except (ValueError, IndexError):
            continue

        numbers = []
        positions = []
        for i in range(4, 4 + num_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            sym = parts[3]
            if sym not in ELEMENT_MAP:
                break
            numbers.append(ELEMENT_MAP[sym])
            positions.append([x, y, z])

        if len(numbers) == num_atoms:
            molecules.append((name, np.array(numbers), np.array(positions)))

        if max_mols and len(molecules) >= max_mols * 2:
            # read more than needed so we can sort and pick smallest
            break

    return molecules


def downsample_mean_2x2x2(arr):
    """Downsample 3D array by 2x2x2 mean pooling."""
    nx, ny, nz = arr.shape
    if nx % 2 or ny % 2 or nz % 2:
        # pad to even
        px = nx % 2
        py = ny % 2
        pz = nz % 2
        arr = np.pad(arr, ((0, px), (0, py), (0, pz)), mode="edge")
        nx, ny, nz = arr.shape
    return arr.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2).mean(axis=(1, 3, 5))


def process_molecule(name, numbers, positions, h, padding_bohr, xc):
    """Run GPAW for one molecule: get v_ion (maxiter=0) and n_pseudo (full SCF)."""
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.center(vacuum=padding_bohr * Bohr)

    # ── Step 1: v_ion (pre-SCF, maxiter=0) ──
    calc_init = GPAW(h=h, mode="fd", xc=xc, maxiter=0, txt=None, spinpol=False)
    atoms.calc = calc_init
    try:
        atoms.get_potential_energy()
    except Exception:
        pass
    v_ion = calc_init.get_effective_potential()
    atoms.calc = None

    # ── Step 2: n_pseudo (full SCF) ──
    calc_scf = GPAW(h=h, mode="fd", xc=xc, txt=None, spinpol=False)
    atoms.calc = calc_scf
    atoms.get_potential_energy()  # full SCF
    n_pseudo = calc_scf.get_pseudo_density()
    atoms.calc = None

    # Shape matching: v_ion might be on a finer grid
    if v_ion.shape != n_pseudo.shape:
        # try 2x downsample
        expected_2x = tuple(2 * s for s in n_pseudo.shape)
        if v_ion.shape == expected_2x:
            v_ion = downsample_mean_2x2x2(v_ion)
        else:
            raise ValueError(
                f"Shape mismatch: v_ion={v_ion.shape}, n_pseudo={n_pseudo.shape}"
            )

    return v_ion, n_pseudo, atoms


def main():
    parser = argparse.ArgumentParser(description="Generate Phase A dataset")
    parser.add_argument("--sdf", default="data/raw/gdb9.sdf")
    parser.add_argument("--out", default="dataset/qm9_1000_phase_a.h5")
    parser.add_argument("--n_mols", type=int, default=1000)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--padding_bohr", type=float, default=4.0)
    parser.add_argument("--xc", default="LDA")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    print(f"Parsing SDF: {args.sdf}")
    all_mols = parse_sdf_molecules(args.sdf, max_mols=None)
    print(f"Parsed {len(all_mols)} molecules")

    # Sort by number of atoms (smallest first)
    all_mols.sort(key=lambda m: len(m[1]))
    selected = all_mols[: args.n_mols]
    print(f"Selected {len(selected)} smallest molecules")
    print(f"  Smallest: {selected[0][0]} ({len(selected[0][1])} atoms)")
    print(f"  Largest:  {selected[-1][0]} ({len(selected[-1][1])} atoms)")

    # Shard selection
    shard_mols = selected[args.shard_index :: args.num_shards]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(shard_mols)} molecules")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.attrs["n_mols"] = args.n_mols
        out.attrs["h"] = args.h
        out.attrs["padding_bohr"] = args.padding_bohr
        out.attrs["xc"] = args.xc

        done = 0
        fail = 0
        for i, (name, numbers, positions) in enumerate(shard_mols):
            t0 = time.time()
            try:
                v_ion, n_pseudo, atoms = process_molecule(
                    name, numbers, positions, args.h, args.padding_bohr, args.xc
                )
                grp = out.create_group(name)
                grp.create_dataset(
                    "v_ion", data=v_ion.astype(np.float16),
                    compression="gzip", compression_opts=4,
                )
                grp.create_dataset(
                    "n_r", data=n_pseudo.astype(np.float16),
                    compression="gzip", compression_opts=4,
                )
                grp.create_dataset("positions", data=positions.astype(np.float32))
                grp.create_dataset("numbers", data=numbers.astype(np.int32))
                grp.attrs["formula"] = atoms.get_chemical_formula()
                grp.attrs["num_atoms"] = len(atoms)
                grp.attrs["cell_angstrom"] = atoms.cell.array
                grp.attrs["grid_spacing_angstrom"] = args.h
                out.flush()
                done += 1
                dt = time.time() - t0
                print(
                    f"[{done}/{len(shard_mols)}] {name} "
                    f"({atoms.get_chemical_formula()}, {len(atoms)} atoms) "
                    f"shape={v_ion.shape} {dt:.1f}s"
                )
            except Exception as exc:
                fail += 1
                print(f"[FAIL] {name}: {exc}")

    print(f"DONE: {done} ok, {fail} failed, output={args.out}")


if __name__ == "__main__":
    main()
