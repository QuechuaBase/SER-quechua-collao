#!/usr/bin/env python3
"""Extract one eGeMAPS functional vector per audio file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm


METADATA_COLUMNS = [
    "utterance_id",
    "audio_path",
    "actor",
    "emotion",
    "arousal",
    "valence",
    "dominance",
    "duration",
]
FEATURE_SETS = ("eGeMAPSv02",)


def flatten_feature_columns(columns: pd.Index) -> list[str]:
    """Convert openSMILE columns to stable CSV-safe names."""
    names = []
    for column in columns:
        if isinstance(column, tuple):
            name = "__".join(str(part) for part in column if str(part))
        else:
            name = str(column)
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("openSMILE returned duplicate feature column names")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract eGeMAPS functionals for every audio in a SER manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--feature_set",
        choices=FEATURE_SETS,
        default="eGeMAPSv02",
    )
    parser.add_argument(
        "--max_failure_rate",
        type=float,
        default=0.20,
        help="Return a non-zero exit status after saving logs if this rate is exceeded.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.max_failure_rate <= 1.0:
        parser.error("--max_failure_rate must be between 0 and 1")
    try:
        import opensmile
    except ImportError as error:
        parser.error(
            "opensmile is not installed; run "
            "'python -m pip install -r requirements_egemaps.txt'"
        )

    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing = sorted(set(METADATA_COLUMNS) - set(manifest.columns))
    if missing:
        parser.error(f"Manifest is missing required columns: {missing}")
    if manifest["utterance_id"].isna().any():
        parser.error("Manifest contains missing utterance_id values")
    if manifest["utterance_id"].duplicated().any():
        parser.error("Manifest contains duplicate utterance_id values")

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, args.feature_set),
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    feature_names: list[str] | None = None

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"Extracting {args.feature_set}",
    ):
        audio_path = Path(str(row["audio_path"])).expanduser()
        if not audio_path.is_absolute():
            audio_path = audio_path.resolve()
        try:
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            extracted = smile.process_file(str(audio_path))
            if len(extracted) != 1:
                raise ValueError(
                    f"Expected one functional vector, received {len(extracted)} rows"
                )
            current_names = flatten_feature_columns(extracted.columns)
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError("openSMILE feature columns changed between audio files")
            values = extracted.iloc[0].to_numpy()
            record = {column: row[column] for column in METADATA_COLUMNS}
            record.update(dict(zip(current_names, values)))
            records.append(record)
        except Exception as error:  # Continue so one corrupt audio does not stop extraction.
            failures.append(
                {
                    "utterance_id": str(row["utterance_id"]),
                    "audio_path": str(row["audio_path"]),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [*METADATA_COLUMNS, *(feature_names or [])]
    pd.DataFrame(records, columns=output_columns).to_csv(output_path, index=False)

    failed_path = output_path.with_name("failed_files.csv")
    pd.DataFrame(
        failures,
        columns=["utterance_id", "audio_path", "error_type", "error"],
    ).to_csv(failed_path, index=False)

    summary = {
        "manifest": str(manifest_path),
        "output_path": str(output_path),
        "number_of_audios_in_manifest": int(len(manifest)),
        "number_of_audios_processed": int(len(records)),
        "number_of_features": int(len(feature_names or [])),
        "failed_audio_count": int(len(failures)),
        "failed_audios": [
            {
                "utterance_id": item["utterance_id"],
                "audio_path": item["audio_path"],
            }
            for item in failures
        ],
        "feature_set": args.feature_set,
        "feature_level": "Functionals",
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "max_failure_rate": args.max_failure_rate,
    }
    summary_path = output_path.with_name("egemaps_features_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    failure_rate = len(failures) / len(manifest) if len(manifest) else 0.0
    print(f"Features: {output_path} ({len(records):,} audio files)")
    print(f"Summary: {summary_path}")
    print(f"Failures: {failed_path} ({len(failures):,})")
    if not records:
        raise RuntimeError("No audio files were processed successfully")
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"Failure rate {failure_rate:.1%} exceeded "
            f"--max_failure_rate={args.max_failure_rate:.1%}"
        )


if __name__ == "__main__":
    main()
