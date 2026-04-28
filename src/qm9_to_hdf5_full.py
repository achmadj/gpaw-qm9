#!/usr/bin/env python3
"""Build full-QM9 raw HDF5 in the same schema as qm9_smallest_20k.h5.

Per molecule group:
  - positions: float32 [N, 3]
  - numbers:   int32   [N]
  - attrs: smiles, num_atoms

Root attrs:
  - description, total_molecules, max_atoms
  - source_root, xyz_files_seen, parse_failed, build_elapsed_seconds
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
KEY_RE = re.compile(r"^(dsgdb9nsd_\d+)$")


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-QM9 raw HDF5 from XYZ files.")
    parser.add_argument(
        "--qm9-root",
        type=str,
        default="/home/achmadjae/gpaw-qm9/dataset/raw/qm9_full_src",
        help="Root directory containing QM9 XYZ files.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="/home/achmadjae/gpaw-qm9/dataset/raw/qm9_full.h5",
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--manifest-out",
        type=str,
        default="/home/achmadjae/gpaw-qm9/dataset/raw/qm9_full_parse_manifest.csv",
        help="CSV manifest path for parse/write status.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into existing output file; skip keys already present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on first parsing/writing error.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of unique molecules to process (0 = all).",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
    )
    parser.add_argument("--compression-opts", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=200)
    return parser.parse_args()


def _compression_kwargs(compression: str, compression_opts: int) -> dict:
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": compression_opts, "shuffle": True}


def _all_xyz_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.xyz") if p.is_file()]


def _pick_better_path(a: Path, b: Path, root: Path) -> Path:
    """Prefer canonical path under dsgdb9nsd.xyz/, then shorter relative path."""
    a_rel = a.relative_to(root)
    b_rel = b.relative_to(root)
    a_canon = a_rel.parts and a_rel.parts[0] == "dsgdb9nsd.xyz"
    b_canon = b_rel.parts and b_rel.parts[0] == "dsgdb9nsd.xyz"
    if a_canon and not b_canon:
        return a
    if b_canon and not a_canon:
        return b
    return a if len(str(a_rel)) <= len(str(b_rel)) else b


def discover_unique_xyz(root: Path) -> list[tuple[str, Path]]:
    """Return sorted unique (key, path), deduplicated by key."""
    by_key: dict[str, Path] = {}
    for p in _all_xyz_files(root):
        stem = p.stem
        m = KEY_RE.match(stem)
        if not m:
            continue
        key = m.group(1)
        if key in by_key:
            by_key[key] = _pick_better_path(by_key[key], p, root)
        else:
            by_key[key] = p
    return sorted(by_key.items(), key=lambda kv: kv[0])


def _safe_float(token: str) -> float:
    return float(token.replace("*^", "e"))


def parse_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, str, int]:
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if len(lines) < 3:
        raise ValueError("XYZ too short")

    num_atoms = int(lines[0].strip())
    if num_atoms <= 0:
        raise ValueError(f"Invalid atom count: {num_atoms}")

    if len(lines) < 2 + num_atoms:
        raise ValueError(f"XYZ atom block incomplete: need {num_atoms} atoms")

    # Keep compatibility with existing raw builder semantics.
    props_tokens = lines[1].split()
    smiles = props_tokens[-1] if props_tokens else ""

    positions = np.empty((num_atoms, 3), dtype=np.float32)
    numbers = np.empty((num_atoms,), dtype=np.int32)

    for i in range(num_atoms):
        tokens = lines[2 + i].split()
        if len(tokens) < 4:
            raise ValueError(f"Malformed atom line at index {i}")
        symbol = tokens[0]
        if symbol.startswith("C"):
            symbol = "C"
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element symbol: {symbol}")
        numbers[i] = ATOMIC_NUMBERS[symbol]
        positions[i, 0] = _safe_float(tokens[1])
        positions[i, 1] = _safe_float(tokens[2])
        positions[i, 2] = _safe_float(tokens[3])

    return positions, numbers, smiles, num_atoms


def iter_rows(items: Iterable[tuple[str, Path]], root: Path):
    for key, path in items:
        yield key, str(path.relative_to(root)), path


def main() -> int:
    args = parse_args()

    qm9_root = Path(args.qm9_root).resolve()
    out_path = Path(args.out_path).resolve()
    manifest_path = Path(args.manifest_out).resolve()

    if not qm9_root.exists():
        raise FileNotFoundError(f"QM9 root not found: {qm9_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    unique_items = discover_unique_xyz(qm9_root)
    all_xyz_seen = len(_all_xyz_files(qm9_root))
    total_unique = len(unique_items)
    if args.limit > 0:
        unique_items = unique_items[: args.limit]

    log(f"QM9 root: {qm9_root}")
    log(f"XYZ files seen (raw): {all_xyz_seen}")
    log(f"Unique molecule keys: {total_unique}")
    log(f"Processing count: {len(unique_items)}")
    log(f"Output H5: {out_path}")
    log(f"Manifest: {manifest_path}")

    mode = "a" if args.resume and out_path.exists() else "w"
    comp_kwargs = _compression_kwargs(args.compression, args.compression_opts)

    started = time.time()
    written = 0
    skipped_existing = 0
    parse_failed = 0
    max_atoms = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["key", "source_relpath", "status", "num_atoms", "error"])

        with h5py.File(out_path, mode) as db:
            if mode == "w":
                db.attrs["description"] = "QM9 full raw molecules parsed from XYZ"
                db.attrs["source_root"] = str(qm9_root)

            for idx, (key, relpath, xyz_path) in enumerate(iter_rows(unique_items, qm9_root), start=1):
                if key in db:
                    skipped_existing += 1
                    writer.writerow([key, relpath, "skip_existing", "", ""])
                    continue

                try:
                    positions, numbers, smiles, num_atoms = parse_xyz(xyz_path)
                    grp = db.create_group(key)
                    grp.create_dataset("positions", data=positions, **comp_kwargs)
                    grp.create_dataset("numbers", data=numbers, **comp_kwargs)
                    grp.attrs["smiles"] = smiles
                    grp.attrs["num_atoms"] = int(num_atoms)
                    written += 1
                    if num_atoms > max_atoms:
                        max_atoms = int(num_atoms)
                    writer.writerow([key, relpath, "ok", int(num_atoms), ""])
                except Exception as exc:
                    parse_failed += 1
                    writer.writerow([key, relpath, "error", "", str(exc)])
                    if args.strict:
                        raise

                if idx % args.flush_every == 0:
                    db.flush()
                    log(
                        f"[{idx}/{len(unique_items)}] written={written} "
                        f"skip={skipped_existing} failed={parse_failed}"
                    )

            db.flush()
            total_groups = len(db.keys())
            if max_atoms == 0 and total_groups > 0:
                max_atoms = max(int(db[k].attrs.get("num_atoms", 0)) for k in db.keys())

            db.attrs["total_molecules"] = int(total_groups)
            db.attrs["max_atoms"] = int(max_atoms)
            db.attrs["xyz_files_seen"] = int(all_xyz_seen)
            db.attrs["unique_molecule_keys"] = int(total_unique)
            db.attrs["parse_failed"] = int(parse_failed)
            db.attrs["build_elapsed_seconds"] = float(time.time() - started)

    elapsed = time.time() - started
    log(
        f"Done. written={written}, skipped_existing={skipped_existing}, "
        f"parse_failed={parse_failed}, elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
