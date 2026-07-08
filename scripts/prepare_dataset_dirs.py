#!/usr/bin/env python3
"""Create the local, Git-ignored dataset directory layout."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SUBDIRS = (
    "raw/scs",
    "raw/sps",
    "raw/add_data",
    "raw/silver",
    "processed/manifests",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create local directories for Puno Quechua datasets."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Dataset root directory (default: data).",
    )
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    for relative in DEFAULT_SUBDIRS:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {directory}")

    print("\nNo datasets were downloaded or modified.")


if __name__ == "__main__":
    main()
