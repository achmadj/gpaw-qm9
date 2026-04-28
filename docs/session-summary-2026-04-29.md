# Session Summary: Revisi Progress Report ke Model v2

**Date:** 2026-04-29 21:30 | **Duration:** ~60 menit | **User:** Achmad Jaelani  
**Environment:** Linux, bash, conda: `3d-unet-qm9` (node19, dual RTX 4090, CUDA 12.6), cwd: `/home/achmadjae/gpaw-qm9/docs`

---

## Project Overview

Project ini adalah riset **Deep Learning untuk DFT (Density Functional Theory)** — memprediksi electron density (`n_r`) dari electrostatic potential (`v_ion`) pada molekul QM9 menggunakan 3D U-Net, untuk **bypass iterasi SCF yang mahal**.

Session ini fokus pada **revisi major `progress_report.tex`**:
1. Melakukan 10 poin revisi konten yang diminta user (hapus teks teknis, ganti grafik jadi tabel, comment-out slide, dll).
2. **Migrasi data dari model v1 (542K params) ke v2 (4.87M params)** — model terbaik yang lebih akurat.
3. Regenerasi semua figure dan tabel dari inferensi v2 via SLURM job di node19.
4. Compile LaTeX, inspeksi visual per slide, dan iterasi perbaikan layout.

---

## Key Decisions Made

- **Decision:** Gunakan **equivariant_v2** (~4.87M params, base_channels=96) sebagai model utama di progress report, bukan v1 (542K).  
  → **Rationale:** v2 secara signifikan lebih akurat (R² 0.9979 vs 0.9935, MAE 0.00460 vs 0.00717). v1 terlalu kecil untuk representasi yang memadai.

- **Decision:** Presentasi diubah menjadi **untuk audiens umum** — hapus semua jargon teknis (`escnn`, `base_channels`, `max_freq`, `order=0`, gdb IDs, dll).  
  → **Rationale:** Progress report adalah presentasi umum, bukan technical deep-dive.

- **Decision:** Slide "Equivariant vs Dense" diubah dari **grafik bar chart** menjadi **tabel compact** R² per molekul.  
  → **Rationale:** User request — tabel lebih informatif dan rapi di slide.

- **Decision:** 3×4 visualization diganti dari molekul training (C₄H₄O, gdb_110) ke **molekul validation (C₆H₄O, gdb_1006)**.  
  → **Rationale:** Konsistensi dengan narasi "validation-only" dan menghindari data leakage.

- **Decision:** Dua slide teknis ("PAW Consistency" dan "Lessons Learned") di-**comment-out**, tidak dihapus.  
  → **Rationale:** User request — supaya bisa dikembalikan nanti jika perlu.

---

## Actions Taken & Results

### Phase 1: Revisi 10 Poin Konten
1. Hapus "(to be upgraded to PBE/HGH)." dari DFT Settings → ✅
2. Hapus "Crucial Fix: Replaced ReLU with Softplus..." dari Architecture → ✅
3. "Masked L1 Loss" → "L1 Loss" → ✅
4. Hapus kolom "ratio integral" dari tabel prediction metrics → ✅
5. Hapus bullet "Conservation of Charge: Integral ratio..." → ✅
6. Simplify methodology block rotasi (hapus `scipy.ndimage.rotate(...)`) → ✅
7. Tabel R²: hapus gdb IDs, sisa rumus molekul saja → ✅
8. Slide "Equivariant vs Dense": grafik → tabel → ✅
9. Slide "GPAW Single-Shot": hapus grafik (kolom kanan) → ✅
10. Comment-out 2 slide (PAW & Bugs) → ✅

### Phase 2: Regenerasi Data v2 (SLURM Job 5147 di node19)
| Task | Script | Output | Key Result |
|------|--------|--------|------------|
| Loss Curve | `run_v2_loss_curve.py` | `loss_vs_epoch.png/pdf` | Best val loss: **0.00377 @ epoch 99** |
| Rotation R² (10 mol) | `run_v2_rotation_r2.py` | stdout log | 90° mean R²=**0.9214**, 180° mean R²=**0.9976** |
| 3×4 Viz | `plot_equivariant_v2_3x4_grid.py` | `equivariant_v2_3x4_scatter.png/svg` | C₆H₄O: orig 0.9986, 90° 0.9884, 180° 0.9985 |
| V2 vs Dense | `run_v2_vs_dense_val.py` | CSV + stdout | Perbandingan 5 validation molecules |
| Pred Metrics (5 mol) | `run_v2_pred_metrics.py` | stdout table | MAE avg=**0.00460**, R² avg=**0.9979** |

### Phase 3: Update LaTeX & Compile
- Update `progress_report.tex` dengan data v2 terbaru.
- Fix path v2 checkpoint di `plot_equivariant_v2_3x4_grid.py` (`/home/achmadjae/models/` → `/home/achmadjae/gpaw-qm9/models/`).
- Compile: `pdflatex` 2× → 10 pages, no errors/warnings.
- Perbaikan layout: slide 6 mengalami overflow, diperbaiki dengan menghapus alertblock dan menyingkat bullets.

---

## Current State / Checkpoint

- **Working directory:** `/home/achmadjae/gpaw-qm9/docs`
- **Git branch:** `master` | **Uncommitted:** yes — banyak file TeX dan script baru yang belum di-commit
- **Last commit:** tidak ada info commit hash (repo gpaw-qm9 ada .git)
- **Key files modified/created:**
  - `/home/achmadjae/gpaw-qm9/docs/progress_report.tex` — **modified** — slide deck final (10 slides, v2 data)
  - `/home/achmadjae/gpaw-qm9/docs/progress_report.pdf` — **modified** — PDF hasil compile
  - `/home/achmadjae/gpaw-qm9/scripts/run_v2_loss_curve.py` — **created** — plot loss curve v2
  - `/home/achmadjae/gpaw-qm9/scripts/run_v2_rotation_r2.py` — **created** — evaluasi rotasi 10 molekul v2
  - `/home/achmadjae/gpaw-qm9/scripts/run_v2_pred_metrics.py` — **created** — metrik prediksi 5 molekul v2
  - `/home/achmadjae/gpaw-qm9/scripts/run_v2_vs_dense_val.py` — **created** — perbandingan v2 vs dense
  - `/home/achmadjae/gpaw-qm9/scripts/plot_equivariant_v2_3x4_grid.py` — **modified** — fix path + ganti ke gdb_1006
  - `/home/achmadjae/gpaw-qm9/jobs/v2_progress_report.sh` — **created** — SLURM job bundling semua evaluasi v2
  - `/home/achmadjae/3d-unet-qm9/figures/loss_vs_epoch.png` — **modified** — loss curve v2 (regenerated)
  - `/home/achmadjae/3d-unet-qm9/figures/equivariant_v2_3x4_scatter.png` — **created** — 3x4 viz C6H4O v2
  - `/home/achmadjae/3d-unet-qm9/figures/equiv_v2_vs_dense_rotation_val_only.csv` — **created** — data perbandingan

---

## Open Questions & Next Steps

### Immediate (next session)
- [ ] **Commit changes** ke git: `progress_report.tex`, script baru, figure baru.
- [ ] Verifikasi apakah user ingin mengganti caption/subtitle di slide tertentu (saat ini masih ada "SO(3)-Equivariant 3D U-Net for GPAW QM9 Dataset" di subtitle).
- [ ] Cek apakah `progress_report.tex` perlu disesuaikan dengan template presentasi BRIN/UI jika ada.

### Short-term (this week)
- [ ] Regenerasi **semua figure lama v1** yang masih ada di repo supaya tidak membingungkan.
- [ ] Buat script evaluasi downstream (HOMO-LUMO injection) dengan v2 untuk melengkapi hasil.
- [ ] Update `docs/README.md` jika ada, untuk merefleksikan perubahan ke v2.

### Long-term / nice-to-have
- [ ] Eksplorasi apakah v2 bisa training lebih lama (>100 epochs) untuk convergence lebih baik.
- [ ] Bandingkan v2 dengan Dense v2 (base_channels=64) untuk evaluasi yang lebih fair.

---

## Important Notes for Next Session

1. **Model terbaik adalah v2** (`equivariant_v2`, base_channels=96, ~4.87M params, checkpoint di `/home/achmadjae/gpaw-qm9/models/experiments/equivariant_v2/checkpoints/best.pt`). JANGAN pakai v1 lagi untuk presentasi/paper.

2. **Environment untuk inferensi/evaluasi:**
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate 3d-unet-qm9
   export LD_LIBRARY_PATH=/home/achmadjae/miniconda3/envs/3d-unet-qm9/lib:$LD_LIBRARY_PATH
   ```

3. **GPU compute di node19** (dual RTX 4090). Submit SLURM job:
   ```bash
   cd /home/achmadjae/gpaw-qm9
   sbatch jobs/v2_progress_report.sh
   ```
   Partition: `dual_4090`, nodelist: `node19`, time limit: 30 menit.

4. **Path v2 checkpoint** di beberapa script lama mungkin masih salah (`/home/achmadjae/models/...` seharusnya `/home/achmadjae/gpaw-qm9/models/...`). Selalu verifikasi sebelum run.

5. **Jangan sebut v1/v2 di presentasi** — cukup "SO(3)-Equivariant 3D U-Net" generik.

---

## File References

| File | Path | Status | Description |
|------|------|--------|-------------|
| progress_report.tex | `/home/achmadjae/gpaw-qm9/docs/progress_report.tex` | modified | Slide deck Beamer final (10 slides, data v2) |
| progress_report.pdf | `/home/achmadjae/gpaw-qm9/docs/progress_report.pdf` | modified | PDF output (2.5M) |
| run_v2_loss_curve.py | `/home/achmadjae/gpaw-qm9/scripts/run_v2_loss_curve.py` | created | Plot loss curve dari v2 history |
| run_v2_rotation_r2.py | `/home/achmadjae/gpaw-qm9/scripts/run_v2_rotation_r2.py` | created | Evaluasi R² 10 molekul acak dengan rotasi |
| run_v2_pred_metrics.py | `/home/achmadjae/gpaw-qm9/scripts/run_v2_pred_metrics.py` | created | Metrik prediksi 5 molekul spesifik |
| run_v2_vs_dense_val.py | `/home/achmadjae/gpaw-qm9/scripts/run_v2_vs_dense_val.py` | created | Perbandingan v2 vs dense (validation-only) |
| plot_equivariant_v2_3x4_grid.py | `/home/achmadjae/gpaw-qm9/scripts/plot_equivariant_v2_3x4_grid.py` | modified | 3x4 viz (fix path + ganti ke gdb_1006) |
| v2_progress_report.sh | `/home/achmadjae/gpaw-qm9/jobs/v2_progress_report.sh` | created | SLURM job bundling evaluasi v2 |
| loss_vs_epoch.png | `/home/achmadjae/3d-unet-qm9/figures/loss_vs_epoch.png` | modified | Loss curve v2 (regenerated) |
| equivariant_v2_3x4_scatter.png | `/home/achmadjae/3d-unet-qm9/figures/equivariant_v2_3x4_scatter.png` | created | 3x4 visualization C6H4O v2 |
| equiv_v2_vs_dense_rotation_val_only.csv | `/home/achmadjae/3d-unet-qm9/figures/equiv_v2_vs_dense_rotation_val_only.csv` | created | Data perbandingan v2 vs dense |
| generate_plots.py | `/home/achmadjae/gpaw-qm9/docs/generate_plots.py` | unmodified | Script lama (plot v1 loss curve) |
| compare_equiv_vs_dense_rotation_val.py | `/home/achmadjae/gpaw-qm9/scripts/compare_equiv_vs_dense_rotation_val.py` | unmodified | Script lama (v1 vs dense) |

---

## Quick Commands

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3d-unet-qm9

# Compile progress report
cd /home/achmadjae/gpaw-qm9/docs
pdflatex -interaction=nonstopmode progress_report.tex

# Submit v2 evaluation job
sbatch /home/achmadjae/gpaw-qm9/jobs/v2_progress_report.sh

# Check job status
squeue -u $USER

# View job output
cat /home/achmadjae/gpaw-qm9/logs/v2_progress_report_*.out
```
