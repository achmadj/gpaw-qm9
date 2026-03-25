import glob
import os
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_xyz(filepath):
    """
    Parse QM9 XYZ file format:
    Line 1: num_atoms
    Line 2: Properties (includes SMILES string)
    Lines 3 to (3 + num_atoms - 1): element_type, x, y, z, partial_charge
    ...
    """
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        # Properties are on line 2, SMILES is the last token
        smiles = lines[1].split()[-1]

        positions = []
        numbers = []
        # Element to atomic number map for QM9
        atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

        for i in range(2, 2 + num_atoms):
            tokens = lines[i].split()
            elem = tokens[0]
            if elem.startswith("C"):
                elem = "C"  # Handle some formatting edge cases if any
            
            numbers.append(atomic_numbers[elem])
            # positions are in Angstrom in QM9 XYZ files
            positions.append([float(x.replace("*^", "e")) for x in tokens[1:4]])

        return {
            "positions": np.array(positions, dtype=np.float32),
            "numbers": np.array(numbers, dtype=np.int32),
            "num_atoms": num_atoms,
            "smiles": smiles,
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def main():
    qm9_dir = str(PROJECT_ROOT / "data" / "raw" / "qm9")
    out_path = str(PROJECT_ROOT / "data" / "raw" / "qm9_smallest_20k.h5")
    top_k = 20000

    # 1. Gather all XYZ files
    print("Finding all QM9 XYZ files...")
    xyz_files = glob.glob(os.path.join(qm9_dir, "*.xyz"))
    if not xyz_files:
        print(f"No XYZ files found in {qm9_dir}")
        return

    print(f"Found {len(xyz_files)} files.")

    # 2. Extract number of atoms for sorting
    # To avoid parsing the whole file (slow), just read the first line
    print("Reading atom counts for sorting...")
    mol_info = []  # (filepath, num_atoms)

    for fp in xyz_files:
        try:
            with open(fp, "r") as f:
                line = f.readline().strip()
                if not line:
                    continue
                n_atoms = int(line)
                mol_info.append((fp, n_atoms))
        except Exception:
            continue

    # 3. Sort by number of atoms ascending
    print("Sorting molecules by size...")
    mol_info.sort(key=lambda x: x[1])

    # 4. Select top K smallest
    smallest_20k = mol_info[:top_k]
    if not smallest_20k:
        print("No molecules selected.")
        return

    max_atoms = smallest_20k[-1][1]
    print(f"Selected {len(smallest_20k)} molecules.")
    print(f"Largest molecule in this subset has {max_atoms} atoms.")

    # 5. Parse and save to HDF5
    print(f"Writing to {out_path}...")
    with h5py.File(out_path, "w") as f:
        for filepath, n_atoms in smallest_20k:
            mol_key = os.path.basename(filepath).replace(".xyz", "")
            data = parse_xyz(filepath)
            if data:
                grp = f.create_group(mol_key)
                grp.create_dataset("positions", data=data["positions"])
                grp.create_dataset("numbers", data=data["numbers"])
                grp.attrs["smiles"] = data["smiles"]
                grp.attrs["num_atoms"] = data["num_atoms"]

    print("Success.")


if __name__ == "__main__":
    main()
