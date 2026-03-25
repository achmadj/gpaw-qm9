import argparse
import glob
import os
import re

import h5py
import numpy as np


ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def formula_to_electrons(formula: str) -> int:
    total = 0
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if symbol in ATOMIC_NUMBERS:
            total += ATOMIC_NUMBERS[symbol] * (int(count) if count else 1)
    return total


def compute_voxel_volume(cell_angstrom, shape) -> float:
    cell = np.asarray(cell_angstrom, dtype=np.float64)
    return float(abs(np.linalg.det(cell)) / np.prod(np.asarray(shape, dtype=np.float64)))


def verify(out_path: str) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"VERIFICATION: {out_path}", flush=True)
    print(f"{'='*60}", flush=True)

    errors = []
    warnings = []

    with h5py.File(out_path, "r") as db:
        keys = sorted(db.keys())
        total = len(keys)
        print(f"[1] Total molecules found : {total}", flush=True)

        # ── 1. Kelengkapan: semua index 1..N hadir ───────────────
        indices = set()
        for k in keys:
            try:
                indices.add(int(k.rsplit("_", 1)[-1]))
            except ValueError:
                warnings.append(f"Key format tidak dikenal: {k}")

        expected = set(range(min(indices), max(indices) + 1))
        missing = expected - indices
        if missing:
            sample = sorted(missing)[:10]
            errors.append(f"[1] {len(missing)} key hilang, contoh: {sample}")
        else:
            print(f"[1] Semua index {min(indices)}–{max(indices)} hadir ✓", flush=True)

        # ── 2. Tiap group punya n_r dan v_ext ────────────────────
        missing_ds = []
        shape_mismatch = []
        for k in keys:
            grp = db[k]
            if "n_r" not in grp or "v_ext" not in grp:
                missing_ds.append(k)
                continue
            if grp["n_r"].shape != grp["v_ext"].shape:
                shape_mismatch.append(
                    f"{k}: n_r={grp['n_r'].shape} v_ext={grp['v_ext'].shape}"
                )

        if missing_ds:
            errors.append(f"[2] {len(missing_ds)} grup tanpa n_r/v_ext: {missing_ds[:5]}")
        else:
            print(f"[2] Semua grup punya n_r dan v_ext ✓", flush=True)

        if shape_mismatch:
            errors.append(f"[2] {len(shape_mismatch)} shape mismatch: {shape_mismatch[:5]}")
        else:
            print(f"[2] Semua shape n_r == v_ext ✓", flush=True)

        # ── 3. Integritas numerik ─────────────────────────────────
        nan_inf_keys = []
        negative_keys = []
        zero_keys = []
        print(f"[3] Memeriksa integritas numerik ({total} molekul)...", flush=True)

        for i, k in enumerate(keys):
            grp = db[k]
            if "n_r" not in grp:
                continue
            n_r = grp["n_r"][...].astype(np.float32)
            v_ext = grp["v_ext"][...].astype(np.float32)

            if not np.isfinite(n_r).all() or not np.isfinite(v_ext).all():
                nan_inf_keys.append(k)
            if (n_r < 0).any():
                negative_keys.append(k)
            if n_r.max() == 0:
                zero_keys.append(k)

            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{total} diperiksa...", flush=True)

        if nan_inf_keys:
            errors.append(f"[3] {len(nan_inf_keys)} molekul NaN/Inf: {nan_inf_keys[:5]}")
        else:
            print(f"[3] Tidak ada NaN/Inf ✓", flush=True)

        if negative_keys:
            errors.append(f"[3] {len(negative_keys)} molekul n_r negatif: {negative_keys[:5]}")
        else:
            print(f"[3] Tidak ada n_r negatif ✓", flush=True)

        if zero_keys:
            errors.append(f"[3] {len(zero_keys)} molekul n_r semua nol: {zero_keys[:5]}")
        else:
            print(f"[3] Tidak ada n_r semua nol ✓", flush=True)

        # ── 4. Konvergensi: total_energy_hartree ada dan nonzero ──
        missing_energy = []
        zero_energy = []
        for k in keys:
            grp = db[k]
            if "total_energy_hartree" not in grp.attrs:
                missing_energy.append(k)
            elif grp.attrs["total_energy_hartree"] == 0.0:
                zero_energy.append(k)

        if missing_energy:
            errors.append(f"[4] {len(missing_energy)} molekul tanpa total_energy_hartree: {missing_energy[:5]}")
        else:
            print(f"[4] Semua molekul punya total_energy_hartree ✓", flush=True)

        if zero_energy:
            warnings.append(f"[4] {len(zero_energy)} molekul energy == 0.0: {zero_energy[:5]}")
        else:
            print(f"[4] Tidak ada energy == 0.0 ✓", flush=True)

    # ── Ringkasan ─────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):", flush=True)
        for w in warnings:
            print(f"  ⚠  {w}", flush=True)

    if errors:
        print(f"ERRORS ({len(errors)}):", flush=True)
        for e in errors:
            print(f"  ✗  {e}", flush=True)
        print(f"\nVerifikasi GAGAL.", flush=True)
        return False
    else:
        print(f"Verifikasi PASSED — {total} molekul, tidak ada error.", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True, help="Glob pattern for shard HDF5 files")
    parser.add_argument("--out_path", required=True, help="Merged HDF5 output path")
    parser.add_argument("--verify", action="store_true", default=False,
                        help="Jalankan verifikasi setelah merge")
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched: {args.input_glob}")

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_groups = 0
    with h5py.File(args.out_path, "w") as out_db:
        out_db.attrs["description"] = "Merged GPAW shard outputs from QM9"
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
                            ds_name,
                            data=dataset[:],
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                        )
                    for attr_name, attr_value in in_group.attrs.items():
                        out_group.attrs[attr_name] = attr_value
                    total_groups += 1

        out_db.attrs["total_molecules"] = total_groups

    print(
        f"Merged {len(shard_paths)} shards into {args.out_path} "
        f"with {total_groups} molecules.",
        flush=True,
    )

    if args.verify:
        ok = verify(args.out_path)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
