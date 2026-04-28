"""Merge pseudo-density shard HDF5 files into a single dataset.

Usage:
    python src/merge_pseudo_shards.py \\
        --input_glob "dataset/shards/gpaw_pseudo_shard_*.h5" \\
        --out_path dataset/gpaw_qm9_pseudo_merged.h5 \\
        --verify
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np


VALENCE_ELECTRONS = {"H": 1, "C": 4, "N": 5, "O": 6, "F": 7}
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def formula_to_valence_electrons(formula: str) -> int:
    total = 0
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if symbol in VALENCE_ELECTRONS:
            total += VALENCE_ELECTRONS[symbol] * (int(count) if count else 1)
    return total


def verify(out_path: str) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"VERIFICATION: {out_path}", flush=True)
    print(f"{'='*60}", flush=True)

    errors = []
    warnings = []

    with h5py.File(out_path, "r") as db:
        keys = sorted(db.keys())
        total = len(keys)
        print(f"[1] Total molecules: {total}", flush=True)

        # Check completeness
        indices = set()
        for k in keys:
            try:
                indices.add(int(k.rsplit("_", 1)[-1]))
            except ValueError:
                warnings.append(f"Unknown key format: {k}")

        print(
            f"[1] Index range: {min(indices)}–{max(indices)} "
            f"(unique IDs: {len(indices)})",
            flush=True,
        )
        # QM9 subsets are not guaranteed to be contiguous in original dataset index.
        # Treat contiguity as informational, not a warning/error condition.

        # Check datasets
        missing_ds = []
        for k in keys:
            grp = db[k]
            if "n_pseudo" not in grp or "v_ext" not in grp:
                missing_ds.append(k)

        if missing_ds:
            errors.append(f"{len(missing_ds)} groups missing n_pseudo/v_ext: {missing_ds[:5]}")
        else:
            print(f"[2] All groups have n_pseudo and v_ext ✓", flush=True)

        # Check numerical integrity
        nan_keys = []
        neg_keys = []
        print(f"[3] Checking numerical integrity ({total} molecules)...", flush=True)
        for i, k in enumerate(keys):
            grp = db[k]
            if "n_pseudo" not in grp:
                continue
            n_p = grp["n_pseudo"][...].astype(np.float32)
            if not np.isfinite(n_p).all():
                nan_keys.append(k)
            if (n_p < 0).any():
                neg_keys.append(k)
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{total} checked...", flush=True)

        if nan_keys:
            errors.append(f"{len(nan_keys)} molecules with NaN/Inf: {nan_keys[:5]}")
        else:
            print(f"[3] No NaN/Inf ✓", flush=True)

        if neg_keys:
            # Pseudo-density can have slightly negative values near boundaries
            warnings.append(f"{len(neg_keys)} molecules with negative n_pseudo: {neg_keys[:5]}")
        else:
            print(f"[3] No negative n_pseudo ✓", flush=True)

        # Check eigenvalues present
        missing_evals = []
        for k in keys:
            if "homo_eV" not in db[k].attrs:
                missing_evals.append(k)

        if missing_evals:
            errors.append(f"{len(missing_evals)} missing eigenvalues: {missing_evals[:5]}")
        else:
            print(f"[4] All molecules have eigenvalues ✓", flush=True)

        # Stats
        homos = np.asarray(
            [db[k].attrs["homo_eV"] for k in keys if "homo_eV" in db[k].attrs],
            dtype=np.float64,
        )
        gaps = np.asarray(
            [db[k].attrs["gap_eV"] for k in keys if "gap_eV" in db[k].attrs],
            dtype=np.float64,
        )
        if homos.size > 0:
            print(f"\n[5] Statistics:", flush=True)
            print(
                f"    HOMO: mean={np.nanmean(homos):.2f}, std={np.nanstd(homos):.2f} eV",
                flush=True,
            )
            finite_gaps = np.isfinite(gaps)
            if finite_gaps.any():
                print(
                    f"    Gap:  mean={np.nanmean(gaps):.2f}, std={np.nanstd(gaps):.2f} eV "
                    f"(finite: {finite_gaps.sum()}/{len(gaps)})",
                    flush=True,
                )
            else:
                warnings.append("All gap_eV values are NaN.")

    # Summary
    print(f"\n{'='*60}", flush=True)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):", flush=True)
        for w in warnings:
            print(f"  ⚠  {w}", flush=True)

    if errors:
        print(f"ERRORS ({len(errors)}):", flush=True)
        for e in errors:
            print(f"  ✗  {e}", flush=True)
        print("Verification FAILED.", flush=True)
        return False
    else:
        print(f"Verification PASSED — {total} molecules, no errors.", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--verify", action="store_true", default=False)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched: {args.input_glob}")

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_groups = 0
    with h5py.File(args.out_path, "w") as out_db:
        out_db.attrs["description"] = "Merged GPAW pseudo-density dataset from QM9"
        out_db.attrs["input_glob"] = args.input_glob
        out_db.attrs["num_shards_found"] = len(shard_paths)

        for shard_path in shard_paths:
            print(f"Merging {shard_path}...", flush=True)
            with h5py.File(shard_path, "r") as shard_db:
                for group_name in shard_db.keys():
                    if group_name in out_db:
                        continue
                    in_group = shard_db[group_name]
                    out_group = out_db.create_group(group_name)
                    for ds_name, dataset in in_group.items():
                        out_group.create_dataset(
                            ds_name, data=dataset[:],
                            compression="gzip", compression_opts=4, shuffle=True,
                        )
                    for attr_name, attr_value in in_group.attrs.items():
                        out_group.attrs[attr_name] = attr_value
                    # Normalize orbital metadata in pseudo-density space.
                    if "formula" in out_group.attrs and "eigenvalues_eV" in out_group.attrs:
                        evals = np.asarray(out_group.attrs["eigenvalues_eV"], dtype=np.float64)
                        if evals.size >= 2:
                            n_val = formula_to_valence_electrons(str(out_group.attrs["formula"]))
                            nocc = max(1, n_val // 2)
                            if nocc >= len(evals):
                                nocc = len(evals) - 1
                            homo = float(evals[nocc - 1])
                            lumo = float(evals[nocc]) if nocc < len(evals) else float("nan")
                            gap = float(lumo - homo) if np.isfinite(lumo) else float("nan")
                            out_group.attrs["n_electrons_valence"] = int(n_val)
                            out_group.attrs["n_occupied"] = np.int32(nocc)
                            out_group.attrs["homo_eV"] = homo
                            out_group.attrs["lumo_eV"] = lumo
                            out_group.attrs["gap_eV"] = gap
                    total_groups += 1

        out_db.attrs["total_molecules"] = total_groups

    print(f"Merged {len(shard_paths)} shards → {args.out_path} "
          f"with {total_groups} molecules.", flush=True)

    if args.verify:
        ok = verify(args.out_path)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
