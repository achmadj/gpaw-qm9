import h5py
import matplotlib.pyplot as plt
import numpy as np

def main():
    h5_path = 'qm9_smallest_20k.h5'
    
    print(f"Loading {h5_path} for analysis...")
    
    atom_counts = []
    # Atomic number to element symbol mapping
    num_to_element = {6: 'C', 1: 'H', 7: 'N', 8: 'O', 9: 'F'}
    element_counts = {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'F': 0}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            molecules = list(f.keys())
            total_mols = len(molecules)
            
            for mol in molecules:
                grp = f[mol]
                
                # 1. Total atoms per molecule
                n_atoms = grp.attrs.get('num_atoms')
                if n_atoms is not None:
                    atom_counts.append(n_atoms)
                
                # 2. Element composition
                numbers = grp['numbers'][:]
                for num in numbers:
                    symbol = num_to_element.get(num, 'Unknown')
                    if symbol in element_counts:
                        element_counts[symbol] += 1
                        
            print(f"Successfully processed {total_mols} molecules.")
            print(f"Minimum atoms in a molecule: {min(atom_counts) if atom_counts else 0}")
            print(f"Maximum atoms in a molecule: {max(atom_counts) if atom_counts else 0}")
            print("Element tallies:")
            for el, count in element_counts.items():
                print(f"  {el}: {count}")
                
    except Exception as e:
        print(f"Error reading HDF5: {e}")
        return

    # Visualization 1: Histogram of Atom Counts
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=range(min(atom_counts), max(atom_counts) + 2), 
             align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title('Distribution of Molecule Sizes in QM9 Sub-Dataset (Top ~20k)', fontsize=14)
    plt.xlabel('Total Number of Atoms (including Hydrogens)', fontsize=12)
    plt.ylabel('Number of Molecules', fontsize=12)
    plt.xticks(range(min(atom_counts), max(atom_counts) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analyze_atom_counts.png', dpi=300)
    print("Saved atom counts histogram to analyze_atom_counts.png")

    # Visualization 2: Element Distribution
    plt.figure(figsize=(10, 6))
    elements = list(element_counts.keys())
    counts = list(element_counts.values())
    colors = ['#4d4d4d', '#e0e0e0', '#1f77b4', '#d62728', '#2ca02c'] # Roughly C,H,N,O,F typical colors
    
    plt.bar(elements, counts, color=colors, edgecolor='black')
    plt.title('Total Elemental Composition across the 20k Subset', fontsize=14)
    plt.xlabel('Element', fontsize=12)
    plt.ylabel('Total Atoms Processed', fontsize=12)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, f'{count:,}', ha='center', fontsize=11)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analyze_elements.png', dpi=300)
    print("Saved element distribution to analyze_elements.png")

if __name__ == "__main__":
    main()
