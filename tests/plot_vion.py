import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW

# 1. Calculate potential for H2O
atoms = Atoms('H2O',
    positions=[(0,0,0), (0.96,0,0), (-0.24,0.93,0)])
atoms.center(vacuum=4.0 * Bohr)

calc = GPAW(h=0.2, mode='fd', xc='LDA', maxiter=0, txt=None)
atoms.calc = calc
try:
    atoms.get_potential_energy()
except Exception:
    pass

vt = calc.get_effective_potential()

# 2. Extract grid indices
nx, ny, nz = vt.shape
x, y, z = np.indices((nx, ny, nz))

# 3. Apply thresholding
# Values are negative, vacuum is near 0. We want regions where potential is strong.
mask = vt < -10.0

x_f = x[mask]
y_f = y[mask]
z_f = z[mask]
vt_f = vt[mask]

# 4. Plot 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map alpha to depth (more negative = more opaque)
alpha_vals = np.clip(np.abs(vt_f) / 50.0, 0.2, 1.0)
colors = np.zeros((len(vt_f), 4))
# Use a colormap and then replace alpha channel
cmap = plt.get_cmap('viridis_r')
norm = plt.Normalize(vmin=vt_f.min(), vmax=vt_f.max())
colors = cmap(norm(vt_f))
colors[:, 3] = alpha_vals

# Color map mapping the potential (most negative is darkest, etc)
sc = ax.scatter(x_f, y_f, z_f, c=colors, s=60)

# Create a ScalarMappable for colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Ionic Potential (eV)')
ax.set_title('Pre-SCF V_ion (maxiter=0) for H2O')
ax.set_xlabel('X grid index')
ax.set_ylabel('Y grid index')
ax.set_zlabel('Z grid index')

# Save figure
plt.savefig('/clusterfs/students/achmadjae/gpaw-qm9/tests/vion_scatter.png', dpi=150, bbox_inches='tight')
print("Plot saved to /clusterfs/students/achmadjae/gpaw-qm9/tests/vion_scatter.png")
