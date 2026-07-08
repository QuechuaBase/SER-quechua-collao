#!/usr/bin/env python3
"""Inspect an extracted Mozilla speech dataset without modifying it."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
TABLE_SUFFIXES = {".tsv", ".csv"}
PATH_COLUMNS = ("path", "audio", "audio_path", "filename", "file", "clip")
SPLIT_COLUMNS = ("split", "subset", "partition", "set")
VALIDATION_COLUMNS = (
    "validated",
    "is_validated",
    "validation",
    "status",
    "up_votes",
    "down_votes",
)


def iter_files(root: Path, suffixes: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def table_dialect(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def inspect_table(path: Path, root: Path) -> dict:
    delimiter = table_dialect(path)
    rows = 0
    columns: list[str] = []
    split_values: Counter[str] = Counter()
    validation_fields: set[str] = set()
    referenced_audio: set[str] = set()

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = reader.fieldnames or []
        split_column = next((name for name in SPLIT_COLUMNS if name in columns), None)
        path_column = next((name for name in PATH_COLUMNS if name in columns), None)
        validation_fields.update(name for name in VALIDATION_COLUMNS if name in columns)
        for row in reader:
            rows += 1
            if split_column and row.get(split_column):
                split_values[row[split_column].strip()] += 1
            if path_column and row.get(path_column):
                referenced_audio.add(row[path_column].strip())

    return {
        "file": str(path.relative_to(root)),
        "rows": rows,
        "columns": columns,
        "split_values": dict(split_values),
        "validation_fields": sorted(validation_fields),
        "referenced_audio_count": len(referenced_audio),
    }


def audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate) if info.samplerate else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect audio, tables, columns, splits, and estimated duration."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument(
        "--skip_duration",
        action="store_true",
        help="Do not open audio headers to estimate total duration.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete report as JSON.",
    )
    args = parser.parse_args()

    root = args.dataset_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory does not exist: {root}")

    audio_files = sorted(iter_files(root, AUDIO_SUFFIXES))
    table_files = sorted(iter_files(root, TABLE_SUFFIXES))
    table_reports = [inspect_table(path, root) for path in table_files]

    measured = 0
    duration_seconds = 0.0
    if not args.skip_duration:
        for path in audio_files:
            duration = audio_duration(path)
            if duration is not None:
                measured += 1
                duration_seconds += duration

    report = {
        "dataset_dir": str(root),
        "audio_file_count": len(audio_files),
        "audio_extensions": dict(Counter(path.suffix.lower() for path in audio_files)),
        "table_file_count": len(table_files),
        "tables": table_reports,
        "estimated_duration_seconds": duration_seconds if measured else None,
        "estimated_duration_hours": duration_seconds / 3600 if measured else None,
        "audio_files_measured": measured,
        "detected_split_files": [
            item["file"]
            for item in table_reports
            if Path(item["file"]).stem.lower()
            in {"train", "dev", "valid", "validation", "test", "validated", "invalidated", "other"}
        ],
        "validation_fields": sorted(
            {field for item in table_reports for field in item["validation_fields"]}
        ),
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"Dataset: {root}")
    print(f"Audio files: {len(audio_files):,}")
    print(f"Audio extensions: {report['audio_extensions'] or 'none'}")
    print(f"TSV/CSV files: {len(table_files):,}")
    if measured:
        print(
            f"Estimated duration: {duration_seconds / 3600:.3f} h "
            f"({measured:,}/{len(audio_files):,} audio headers read)"
        )
    elif args.skip_duration:
        print("Estimated duration: skipped")
    else:
        print("Estimated duration: unavailable (install soundfile or check audio format)")

    for item in table_reports:
        print(f"\nTable: {item['file']}")
        print(f"  Rows/clips: {item['rows']:,}")
        print(f"  Columns: {', '.join(item['columns']) or 'none'}")
        print(f"  Referenced audio: {item['referenced_audio_count']:,}")
        if item["split_values"]:
            print(f"  Split values: {item['split_values']}")
        if item["validation_fields"]:
            print(f"  Validation fields: {', '.join(item['validation_fields'])}")

    print("\nDetected split files:", ", ".join(report["detected_split_files"]) or "none")
    print("Validation fields:", ", ".join(report["validation_fields"]) or "none")


if __name__ == "__main__":
    main()
