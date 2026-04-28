#!/usr/bin/env python3
"""Validate H5 pseudo-density injection vs full SCF for all HGH/PBE shards.

For each shard file, randomly sample N molecule groups and compare:
1) Full SCF HOMO/LUMO from GPAW
2) Fixed-density single-shot HOMO/LUMO after injecting stored n_pseudo

Important:
- Stored n_pseudo is from get_pseudo_density(pad=True), so mapping to nt_sG
  must account for possible +1 padded axes.
- We intentionally avoid atoms.get_potential_energy() after injection to
  prevent SCF reinitialization from overwriting injected density.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import time
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW


# Prevent SLURM MPI context leakage in serial script mode.
for _k in list(os.environ.keys()):
    if _k.startswith(("PMIX_", "PMI_", "SLURM_MPI_", "SLURM_PMIX")):
        os.environ.pop(_k, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate shard H5 n_pseudo injection against full SCF HOMO/LUMO."
    )
    parser.add_argument(
        "--raw_db",
        default="/home/achmadjae/gpaw-qm9/dataset/raw/qm9_smallest_20k.h5",
        help="Path to raw QM9 geometry database.",
    )
    parser.add_argument(
        "--shard_glob",
        default="/home/achmadjae/gpaw-qm9/dataset/shards/gpaw_pseudo_hgh_pbe_shard_*.h5",
        help="Glob for shard files to validate.",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=2,
        help="Random molecules sampled per shard.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--h", type=float, default=0.2, help="GPAW grid spacing (Angstrom).")
    parser.add_argument("--xc", type=str, default="PBE", help="GPAW XC functional.")
    parser.add_argument("--mode", type=str, default="fd", help="GPAW mode.")
    parser.add_argument("--setups", type=str, default="hgh", help="GPAW setups mode.")
    parser.add_argument("--padding_bohr", type=float, default=4.0, help="Vacuum padding in Bohr.")
    parser.add_argument(
        "--max_diag_iters",
        type=int,
        default=12,
        help="Max fixed-Hamiltonian eigensolver iterations.",
    )
    parser.add_argument(
        "--eig_stability_tol",
        type=float,
        default=1e-6,
        help="Stop eigensolver iterations when max|delta eig| below this value (eV).",
    )
    parser.add_argument(
        "--frontier_tol_eV",
        type=float,
        default=0.05,
        help="Pass threshold for |dH|, |dL|, and |dGap| in eV.",
    )
    parser.add_argument(
        "--out_csv",
        default="/home/achmadjae/gpaw-qm9/logs/validate_hgh_pbe_2per_shard.csv",
        help="Output CSV path for per-molecule validation rows.",
    )
    parser.add_argument(
        "--out_summary_json",
        default="/home/achmadjae/gpaw-qm9/logs/validate_hgh_pbe_2per_shard_summary.json",
        help="Output JSON path for aggregate summary.",
    )
    return parser.parse_args()


def parse_shard_index(path: str) -> int:
    match = re.search(r"_shard_(\d+)\.h5$", path)
    return int(match.group(1)) if match else -1


def axis_slice(src_n: int, dst_n: int) -> slice:
    if src_n == dst_n:
        return slice(0, dst_n)
    if src_n == dst_n + 1:
        return slice(1, 1 + dst_n)
    if src_n > dst_n:
        start = (src_n - dst_n) // 2
        return slice(start, start + dst_n)
    raise ValueError(f"Cannot map source axis {src_n} to destination axis {dst_n}")


def inject_pseudo_density(calc: GPAW, n_pseudo_h5: np.ndarray) -> None:
    nx, ny, nz = calc.density.nt_sG.shape[1:]
    sx = axis_slice(int(n_pseudo_h5.shape[0]), nx)
    sy = axis_slice(int(n_pseudo_h5.shape[1]), ny)
    sz = axis_slice(int(n_pseudo_h5.shape[2]), nz)
    n_internal = n_pseudo_h5[sx, sy, sz] * (Bohr ** 3)
    if n_internal.shape != (nx, ny, nz):
        raise ValueError(
            f"Mapped density shape {n_internal.shape} != target {(nx, ny, nz)}"
        )
    calc.density.nt_sG[0, :] = n_internal
    calc.density.interpolate_pseudo_density()
    calc.density.calculate_pseudo_charge()
    calc.hamiltonian.update(calc.density)


def run_fixed_density_eigs(calc: GPAW, max_iters: int, stability_tol: float) -> np.ndarray:
    prev = None
    for _ in range(max_iters):
        calc.wfs.eigensolver.iterate(calc.hamiltonian, calc.wfs)
        eig = calc.get_eigenvalues(spin=0)
        if prev is not None and np.max(np.abs(eig - prev)) < stability_tol:
            break
        prev = eig.copy()
    return calc.get_eigenvalues(spin=0)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched: {args.shard_glob}")

    ensure_parent(args.out_csv)
    ensure_parent(args.out_summary_json)

    rows = []
    shard_summaries = []
    started = time.time()

    print(f"Found {len(shard_paths)} shard files.")
    print(
        f"Config: samples_per_shard={args.samples_per_shard}, "
        f"xc={args.xc}, mode={args.mode}, setups={args.setups}, h={args.h}"
    )

    for shard_i, shard_path in enumerate(shard_paths, start=1):
        shard_idx = parse_shard_index(shard_path)
        with h5py.File(shard_path, "r") as sf:
            keys = sorted(sf.keys())

        if not keys:
            shard_summaries.append(
                {
                    "shard_index": shard_idx,
                    "shard_path": shard_path,
                    "available_keys": 0,
                    "sampled": 0,
                    "success": 0,
                    "failed": 0,
                    "pass_count": 0,
                }
            )
            continue

        per_shard_rng = random.Random(args.seed + max(shard_idx, 0))
        sample_n = min(args.samples_per_shard, len(keys))
        sampled_keys = per_shard_rng.sample(keys, sample_n)

        shard_success = 0
        shard_failed = 0
        shard_pass = 0

        print(
            f"[{shard_i}/{len(shard_paths)}] shard={shard_idx} "
            f"keys={len(keys)} sampled={sample_n}"
        )

        for key in sampled_keys:
            t0 = time.time()
            row = {
                "shard_index": shard_idx,
                "shard_file": os.path.basename(shard_path),
                "key": key,
                "status": "fail",
                "formula": "",
                "delta_homo_eV": np.nan,
                "delta_lumo_eV": np.nan,
                "delta_gap_eV": np.nan,
                "max_abs_eig_delta_eV": np.nan,
                "pass_frontier_tol": False,
                "error": "",
                "elapsed_s": np.nan,
            }

            try:
                with h5py.File(args.raw_db, "r") as raw_db, h5py.File(shard_path, "r") as sf:
                    positions = raw_db[key]["positions"][:]
                    numbers = raw_db[key]["numbers"][:]
                    n_h5 = sf[key]["n_pseudo"][:].astype(np.float64)
                    formula = str(sf[key].attrs.get("formula", "unknown"))

                row["formula"] = formula

                atoms = Atoms(numbers=numbers, positions=positions)
                atoms.center(vacuum=args.padding_bohr * Bohr)

                # Full SCF reference.
                calc_full = GPAW(
                    h=args.h,
                    mode=args.mode,
                    xc=args.xc,
                    setups=args.setups,
                    txt=None,
                    spinpol=False,
                )
                atoms.calc = calc_full
                atoms.get_potential_energy()
                occ = int(np.sum(calc_full.get_occupation_numbers(spin=0) > 0.5))
                nbands = calc_full.get_number_of_bands()
                eig_full = calc_full.get_eigenvalues(spin=0)
                homo_full = float(eig_full[occ - 1])
                lumo_full = float(eig_full[occ])
                atoms.calc = None

                # Fixed-density injection run.
                atoms_inj = atoms.copy()
                calc_inj = GPAW(
                    h=args.h,
                    mode=args.mode,
                    xc=args.xc,
                    setups=args.setups,
                    txt=None,
                    spinpol=False,
                    nbands=nbands,
                )
                atoms_inj.calc = calc_inj
                calc_inj.initialize(atoms_inj)
                calc_inj.set_positions(atoms_inj)
                inject_pseudo_density(calc_inj, n_h5)
                eig_inj = run_fixed_density_eigs(
                    calc_inj, max_iters=args.max_diag_iters, stability_tol=args.eig_stability_tol
                )

                homo_inj = float(eig_inj[occ - 1])
                lumo_inj = float(eig_inj[occ])
                gap_full = lumo_full - homo_full
                gap_inj = lumo_inj - homo_inj

                d_h = abs(homo_inj - homo_full)
                d_l = abs(lumo_inj - lumo_full)
                d_g = abs(gap_inj - gap_full)

                ncmp = min(len(eig_full), len(eig_inj))
                max_eig = float(np.max(np.abs(eig_inj[:ncmp] - eig_full[:ncmp])))
                pass_flag = (
                    d_h <= args.frontier_tol_eV
                    and d_l <= args.frontier_tol_eV
                    and d_g <= args.frontier_tol_eV
                )

                row.update(
                    {
                        "status": "ok",
                        "delta_homo_eV": d_h,
                        "delta_lumo_eV": d_l,
                        "delta_gap_eV": d_g,
                        "max_abs_eig_delta_eV": max_eig,
                        "pass_frontier_tol": pass_flag,
                        "elapsed_s": time.time() - t0,
                    }
                )
                shard_success += 1
                if pass_flag:
                    shard_pass += 1
            except Exception as exc:  # keep per-sample failures visible, continue next sample
                row["error"] = str(exc)
                row["elapsed_s"] = time.time() - t0
                shard_failed += 1

            rows.append(row)

        shard_summaries.append(
            {
                "shard_index": shard_idx,
                "shard_path": shard_path,
                "available_keys": len(keys),
                "sampled": sample_n,
                "success": shard_success,
                "failed": shard_failed,
                "pass_count": shard_pass,
            }
        )

    # Write CSV.
    fields = [
        "shard_index",
        "shard_file",
        "key",
        "formula",
        "status",
        "delta_homo_eV",
        "delta_lumo_eV",
        "delta_gap_eV",
        "max_abs_eig_delta_eV",
        "pass_frontier_tol",
        "elapsed_s",
        "error",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    fail_rows = [r for r in rows if r["status"] != "ok"]

    def stat(values: list[float], fn, default=np.nan) -> float:
        return float(fn(values)) if values else float(default)

    dh = [float(r["delta_homo_eV"]) for r in ok_rows]
    dl = [float(r["delta_lumo_eV"]) for r in ok_rows]
    dg = [float(r["delta_gap_eV"]) for r in ok_rows]

    summary = {
        "config": {
            "raw_db": args.raw_db,
            "shard_glob": args.shard_glob,
            "samples_per_shard": args.samples_per_shard,
            "seed": args.seed,
            "h": args.h,
            "xc": args.xc,
            "mode": args.mode,
            "setups": args.setups,
            "padding_bohr": args.padding_bohr,
            "max_diag_iters": args.max_diag_iters,
            "eig_stability_tol": args.eig_stability_tol,
            "frontier_tol_eV": args.frontier_tol_eV,
        },
        "timing_s": time.time() - started,
        "counts": {
            "shards": len(shard_paths),
            "rows_total": len(rows),
            "rows_ok": len(ok_rows),
            "rows_failed": len(fail_rows),
            "rows_pass_frontier_tol": int(sum(bool(r["pass_frontier_tol"]) for r in ok_rows)),
        },
        "metrics_eV": {
            "delta_homo_mean": stat(dh, np.mean),
            "delta_homo_median": stat(dh, np.median),
            "delta_homo_max": stat(dh, np.max),
            "delta_lumo_mean": stat(dl, np.mean),
            "delta_lumo_median": stat(dl, np.median),
            "delta_lumo_max": stat(dl, np.max),
            "delta_gap_mean": stat(dg, np.mean),
            "delta_gap_median": stat(dg, np.median),
            "delta_gap_max": stat(dg, np.max),
        },
        "per_shard": shard_summaries,
    }

    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Validation rows written: {args.out_csv}")
    print(f"Validation summary written: {args.out_summary_json}")
    print(
        f"OK/total: {summary['counts']['rows_ok']}/{summary['counts']['rows_total']}, "
        f"pass(frontier_tol={args.frontier_tol_eV}eV): {summary['counts']['rows_pass_frontier_tol']}"
    )
    print(
        "Mean |dH|, |dL|, |dGap| (eV): "
        f"{summary['metrics_eV']['delta_homo_mean']:.6f}, "
        f"{summary['metrics_eV']['delta_lumo_mean']:.6f}, "
        f"{summary['metrics_eV']['delta_gap_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
