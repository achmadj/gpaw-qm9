#!/usr/bin/env python3
import argparse
import os
import sys

import h5py


def format_shape(shape):
    return "x".join(str(x) for x in shape) if shape else "scalar"


def format_value(value):
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return repr(value)
    return repr(value)


def inspect_group(group, group_name, show_attrs, max_datasets):
    print(f"\n[group] {group_name}")

    datasets = []
    subgroups = []

    for name, item in group.items():
        if isinstance(item, h5py.Dataset):
            datasets.append((name, item))
        elif isinstance(item, h5py.Group):
            subgroups.append(name)

    print(f"  datasets: {len(datasets)}")
    print(f"  subgroups: {len(subgroups)}")

    if datasets:
        print("  dataset details:")
        for idx, (ds_name, ds) in enumerate(datasets):
            if idx >= max_datasets:
                remaining = len(datasets) - max_datasets
                print(f"    ... ({remaining} more datasets not shown)")
                break

            compression = ds.compression or "none"
            print(
                f"    - {ds_name}: "
                f"shape={format_shape(ds.shape)}, "
                f"dtype={ds.dtype}, "
                f"compression={compression}"
            )

    if show_attrs:
        attrs = list(group.attrs.items())
        print(f"  attributes: {len(attrs)}")
        for key, value in attrs:
            print(f"    - {key} = {format_value(value)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect contents of a GPAW shard HDF5 file."
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="gpaw_qm9_shard_0.h5",
        help="Path to the HDF5 file to inspect (default: gpaw_qm9_shard_0.h5)",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=5,
        help="Maximum number of top-level groups to inspect in detail",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=10,
        help="Maximum number of datasets to show per group",
    )
    parser.add_argument(
        "--no-attrs",
        action="store_true",
        help="Do not print group attributes",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print summary and top-level group names",
    )
    args = parser.parse_args()

    h5_path = args.h5_path

    if not os.path.exists(h5_path):
        print(f"ERROR: file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting HDF5 file: {h5_path}")
    print(f"File size: {os.path.getsize(h5_path)} bytes")

    with h5py.File(h5_path, "r") as f:
        top_level_keys = list(f.keys())

        print("\n[file attributes]")
        if len(f.attrs) == 0:
            print("  (none)")
        else:
            for key, value in f.attrs.items():
                print(f"  - {key} = {format_value(value)}")

        print("\n[summary]")
        print(f"  top-level groups: {len(top_level_keys)}")

        if top_level_keys:
            print("  first group names:")
            for name in top_level_keys[: min(10, len(top_level_keys))]:
                print(f"    - {name}")

        if args.list_only:
            return

        groups_to_show = top_level_keys[: args.max_groups]
        print(f"\nInspecting up to {len(groups_to_show)} groups in detail...")

        for group_name in groups_to_show:
            obj = f[group_name]
            if isinstance(obj, h5py.Group):
                inspect_group(
                    obj,
                    group_name=group_name,
                    show_attrs=not args.no_attrs,
                    max_datasets=args.max_datasets,
                )
            else:
                print(f"\n[non-group object] {group_name}: type={type(obj).__name__}")

    print("\nDone.")


if __name__ == "__main__":
    main()
