#!/usr/bin/env python3
"""Generate bar chart figure for GPAW rotation injection comparison."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Load report
report_path = Path("/home/achmadjae/gpaw-qm9/logs/gdb1006_lda_rotation_report.json")
fig_dir = Path("/home/achmadjae/3d-unet-qm9/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

with open(report_path) as f:
    report = json.load(f)

cases = report["cases"]
comps = report["comparisons"]

labels = ["Original", "90° (y-axis)", "180° (y-axis)"]
full_labels = ["original_full_scf", "rotated_90_y_full_scf", "rotated_180_y_full_scf"]
inj_labels = ["original_injected", "rotated_90_y_injected", "rotated_180_y_injected"]

# Extract data
full_e = [cases[k]["e_total_free_eV"] for k in full_labels]
inj_e = [cases[k]["e_total_free_eV"] for k in inj_labels]
full_homo = [cases[k]["homo_eV"] for k in full_labels]
inj_homo = [cases[k]["homo_eV"] for k in inj_labels]
full_lumo = [cases[k]["lumo_eV"] for k in full_labels]
inj_lumo = [cases[k]["lumo_eV"] for k in inj_labels]
full_gap = [cases[k]["gap_eV"] for k in full_labels]
inj_gap = [cases[k]["gap_eV"] for k in inj_labels]

# Delta energies
delta_e = [comps[k].get("delta_e_total_free_eV", 0) for k in ["original", "rotated_90_y", "rotated_180_y"]]
delta_homo = [comps[k].get("delta_homo_eV", 0) for k in ["original", "rotated_90_y", "rotated_180_y"]]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("GPAW Single-Shot Injection Under Rotation — gdb\\_1006 (C$_6$H$_4$O, LDA)", fontsize=14, fontweight='bold')

x = np.arange(len(labels))
width = 0.35

# Panel 1: Total energy
ax = axes[0, 0]
bars1 = ax.bar(x - width/2, full_e, width, label="Full SCF", color="#2ca02c", edgecolor='black')
bars2 = ax.bar(x + width/2, inj_e, width, label="Injected Single-Shot", color="#ff7f0e", edgecolor='black')
ax.set_ylabel("$E_{\\text{total}}$ (eV)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_title("(a) Total Energy", fontsize=12)
# Annotate deltas
for i, de in enumerate(delta_e):
    ax.annotate(f"$\\Delta$={de:+.2e}", xy=(x[i], min(full_e[i], inj_e[i]) - 0.5),
                ha='center', fontsize=8, color='red')

# Panel 2: HOMO
ax = axes[0, 1]
bars1 = ax.bar(x - width/2, full_homo, width, label="Full SCF", color="#2ca02c", edgecolor='black')
bars2 = ax.bar(x + width/2, inj_homo, width, label="Injected Single-Shot", color="#ff7f0e", edgecolor='black')
ax.set_ylabel("HOMO (eV)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_title("(b) HOMO Energy", fontsize=12)
for i, dh in enumerate(delta_homo):
    ax.annotate(f"$\\Delta$={dh:+.3f}", xy=(x[i], min(full_homo[i], inj_homo[i]) - 0.3),
                ha='center', fontsize=8, color='red')

# Panel 3: LUMO
ax = axes[1, 0]
bars1 = ax.bar(x - width/2, full_lumo, width, label="Full SCF", color="#2ca02c", edgecolor='black')
bars2 = ax.bar(x + width/2, inj_lumo, width, label="Injected Single-Shot", color="#ff7f0e", edgecolor='black')
ax.set_ylabel("LUMO (eV)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_title("(c) LUMO Energy", fontsize=12)

# Panel 4: Gap
ax = axes[1, 1]
bars1 = ax.bar(x - width/2, full_gap, width, label="Full SCF", color="#2ca02c", edgecolor='black')
bars2 = ax.bar(x + width/2, inj_gap, width, label="Injected Single-Shot", color="#ff7f0e", edgecolor='black')
ax.set_ylabel("Gap (eV)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_title("(d) HOMO-LUMO Gap", fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_path = fig_dir / "gpaw_rotation_injection_comparison.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
print(f"Saved: {out_path}")
print(f"Saved: {out_path.with_suffix('.pdf')}")

# Also create a focused delta-only figure
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, [abs(de) for de in delta_e], width, label="$|\\Delta E_{\\text{total}}|$ (eV)", color="#d62728", edgecolor='black')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, [abs(dh) for dh in delta_homo], width, label="$|\\Delta \\varepsilon_{\\text{HOMO}}|$ (eV)", color="#1f77b4", edgecolor='black')

ax.set_yscale('log')
ax.set_ylabel("$|\\Delta E_{\\text{total}}|$ (eV)", fontsize=12, color="#d62728")
ax2.set_ylabel("$|\\Delta \\varepsilon_{\\text{HOMO}}|$ (eV)", fontsize=12, color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Single-Shot Injection Error vs Full SCF", fontsize=13, fontweight='bold')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Annotate exact values
for i in range(len(labels)):
    ax.annotate(f"{abs(delta_e[i]):.2e}", xy=(x[i] - width/2, abs(delta_e[i])),
                ha='center', va='bottom', fontsize=9, color='#d62728', fontweight='bold')
    ax2.annotate(f"{abs(delta_homo[i]):.3f}", xy=(x[i] + width/2, abs(delta_homo[i])),
                 ha='center', va='bottom', fontsize=9, color='#1f77b4', fontweight='bold')

plt.tight_layout()
out_path2 = fig_dir / "gpaw_rotation_injection_delta.png"
plt.savefig(out_path2, dpi=300, bbox_inches='tight')
plt.savefig(out_path2.with_suffix('.pdf'), bbox_inches='tight')
print(f"Saved: {out_path2}")
print(f"Saved: {out_path2.with_suffix('.pdf')}")
