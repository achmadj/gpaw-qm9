import h5py

try:
    with h5py.File('qm9_smallest_20k.h5', 'r') as f:
        print(f"Total molecules saved: {len(f.keys())}")
        print(f"Max atoms attribute: {f.attrs.get('max_atoms', 'N/A')}")
        
        mols = list(f.keys())
        if mols:
            first_mol = mols[0]
            last_mol = mols[-1]
            print(f"First molecule: {first_mol} - atoms: {f[first_mol].attrs.get('num_atoms')} - SMILES: {f[first_mol].attrs.get('smiles')}")
            print(f"Last molecule: {last_mol} - atoms: {f[last_mol].attrs.get('num_atoms')} - SMILES: {f[last_mol].attrs.get('smiles')}")
except Exception as e:
    print(f"Error reading HDF5: {e}")
