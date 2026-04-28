from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW
import numpy as np

atoms = Atoms('H2O',
    positions=[(0,0,0), (0.96,0,0), (-0.24,0.93,0)])
atoms.center(vacuum=4.0 * Bohr)
calc = GPAW(h=0.2, mode='fd', xc='LDA', maxiter=0, txt=None)
atoms.calc = calc
try:
    atoms.get_potential_energy()
except Exception:
    pass

try:
    vt = calc.get_effective_potential()
    print(f"SUCCESS: v_ion shape = {vt.shape}")
    print(f"min={vt.min():.4f}, max={vt.max():.4f}")
except Exception as e:
    print(f"get_effective_potential failed: {e}")
    try:
        vt = calc.hamiltonian.vt_sG[0]
        print(f"SUCCESS (internal): shape = {vt.shape}")
    except Exception as e2:
        print(f"Internal access also failed: {e2}")
        print("Try: calc.dft.potential.vt_sR")
