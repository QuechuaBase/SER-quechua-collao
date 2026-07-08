#!/usr/bin/env python3
"""Explicitly download selected files from a Hugging Face model repository."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PATTERNS = ["checkpoint_best.pt", "*.json", "*.txt", "*.model"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ASR checkpoint files from Hugging Face on explicit request."
    )
    parser.add_argument(
        "--repo_id",
        default="QuechuaBase/xls-r-cpt-qxp-silver",
        help="Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/checkpoints/xls-r-cpt-qxp-silver"),
        help="Local destination directory.",
    )
    parser.add_argument(
        "--allow_patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Only download matching files.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit revision.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        parser.error(
            "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        )
        raise SystemExit(2) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repository: {args.repo_id}")
    print(f"Allowed files: {', '.join(args.allow_patterns)}")
    print(f"Destination: {output_dir}")

    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=args.allow_patterns,
    )
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    print(f"\nDownload completed: {downloaded_path}")
    if files:
        print("Files:")
        for path in files:
            print(f"  - {path.relative_to(output_dir)}")
    else:
        print("WARNING: no files matched the requested patterns.")


if __name__ == "__main__":
    main()
