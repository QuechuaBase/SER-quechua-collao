#!/usr/bin/env python3
"""Build normalized ASR manifests from extracted Mozilla/Common Voice data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TABLE_SUFFIXES = {".tsv", ".csv"}
PATH_COLUMNS = ("path", "audio", "audio_path", "filename", "file", "clip")
TEXT_COLUMNS = ("sentence", "transcription", "text", "transcript")
CLIENT_COLUMNS = ("client_id", "speaker_id", "speaker", "user_id")
SPLIT_COLUMNS = ("split", "subset", "partition", "set")
TRUE_VALUES = {"1", "true", "yes", "y", "validated", "valid", "approved"}
OUTPUT_FIELDS = (
    "path",
    "transcription",
    "source",
    "split",
    "validated",
    "client_id",
)


def first_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def infer_split(path: Path, row: dict[str, str], split_column: str | None) -> str:
    if split_column and row.get(split_column):
        value = row[split_column].strip().lower()
    else:
        value = path.stem.lower()
    aliases = {
        "dev": "valid",
        "validation": "valid",
        "val": "valid",
        "train_silver": "silver",
    }
    value = aliases.get(value, value)
    for name in ("silver", "validated", "train", "valid", "test"):
        if name in value:
            return name
    return "unspecified"


def infer_validated(
    path: Path, row: dict[str, str], split: str, columns: list[str]
) -> bool:
    for name in ("validated", "is_validated", "validation", "status"):
        if name in columns and row.get(name):
            return row[name].strip().lower() in TRUE_VALUES
    if "up_votes" in columns:
        try:
            up = int(row.get("up_votes") or 0)
            down = int(row.get("down_votes") or 0)
            return up > down
        except ValueError:
            pass
    return split in {"train", "valid", "test", "validated"} or "validated" in path.stem.lower()


def resolve_audio_path(dataset_root: Path, table: Path, value: str) -> str:
    supplied = Path(value)
    candidates = [
        supplied if supplied.is_absolute() else dataset_root / supplied,
        table.parent / supplied,
        dataset_root / "clips" / supplied.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return candidate.resolve().relative_to(dataset_root).as_posix()
            except ValueError:
                return str(candidate.resolve())
    return supplied.as_posix()


def read_records(dataset_root: Path, table: Path) -> tuple[list[dict[str, str]], list[str]]:
    delimiter = "\t" if table.suffix.lower() == ".tsv" else ","
    records: list[dict[str, str]] = []
    warnings: list[str] = []
    with table.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = reader.fieldnames or []
        path_column = first_column(columns, PATH_COLUMNS)
        text_column = first_column(columns, TEXT_COLUMNS)
        client_column = first_column(columns, CLIENT_COLUMNS)
        split_column = first_column(columns, SPLIT_COLUMNS)
        if not path_column or not text_column:
            return [], [
                f"Skipped {table}: needs one path column {PATH_COLUMNS} and "
                f"one text column {TEXT_COLUMNS}; found {columns}"
            ]
        for row in reader:
            audio_value = (row.get(path_column) or "").strip()
            text = " ".join((row.get(text_column) or "").split())
            if not audio_value or not text:
                continue
            split = infer_split(table, row, split_column)
            records.append(
                {
                    "path": resolve_audio_path(dataset_root, table, audio_value),
                    "transcription": text,
                    "source": dataset_root.name,
                    "split": split,
                    "validated": str(
                        infer_validated(table, row, split, columns)
                    ).lower(),
                    "client_id": (row.get(client_column) or "").strip()
                    if client_column
                    else "",
                }
            )
    return records, warnings


def write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def deduplicate(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        unique.setdefault((record["source"], record["path"]), record)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean train/valid/test/validated/silver manifests for QXP ASR."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more extracted SCS, SPS, Add_data, or silver directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/manifests"),
    )
    parser.add_argument(
        "--include_tables",
        nargs="*",
        help="Optional table basenames to include, e.g. train.tsv validated.tsv.",
    )
    args = parser.parse_args()

    roots = [path.expanduser().resolve() for path in args.dataset_dir]
    for root in roots:
        if not root.is_dir():
            parser.error(f"Dataset directory does not exist: {root}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    warnings: list[str] = []
    tables_read: list[str] = []
    include = set(args.include_tables or [])
    for root in roots:
        for table in sorted(root.rglob("*")):
            if not table.is_file() or table.suffix.lower() not in TABLE_SUFFIXES:
                continue
            if include and table.name not in include:
                continue
            table_records, table_warnings = read_records(root, table)
            if table_records:
                tables_read.append(str(table))
                records.extend(table_records)
            warnings.extend(table_warnings)

    records = deduplicate(records)
    split_records = {
        name: [record for record in records if record["split"] == name]
        for name in ("train", "valid", "test", "silver")
    }
    validated_records = [
        record for record in records if record["validated"] == "true"
    ]

    for name in ("train", "valid", "test"):
        write_manifest(output_dir / f"{name}.tsv", split_records[name])
    write_manifest(output_dir / "validated.tsv", validated_records)
    silver_path = output_dir / "silver.tsv"
    if split_records["silver"]:
        write_manifest(silver_path, split_records["silver"])
    elif silver_path.exists():
        silver_path.unlink()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dirs": [str(root) for root in roots],
        "tables_read": tables_read,
        "total_unique_records": len(records),
        "manifest_counts": {
            "train": len(split_records["train"]),
            "valid": len(split_records["valid"]),
            "test": len(split_records["test"]),
            "validated": len(validated_records),
            "silver": len(split_records["silver"]),
            "unspecified": sum(record["split"] == "unspecified" for record in records),
        },
        "source_counts": dict(Counter(record["source"] for record in records)),
        "columns": list(OUTPUT_FIELDS),
        "warnings": warnings,
        "notes": [
            "Paths are relative to each source dataset when the audio file was found.",
            "No audio was copied, converted, or downloaded.",
            "validated.tsv is an aggregate view and can overlap train/valid/test.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Output directory: {output_dir}")
    for name, count in metadata["manifest_counts"].items():
        suffix = ".tsv" if name != "unspecified" else ""
        print(f"{name}{suffix}: {count:,}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
